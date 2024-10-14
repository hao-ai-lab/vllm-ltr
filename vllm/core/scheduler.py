import enum, os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import Policy, PolicyFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from vllm.utils import merge_dicts
import numpy as np
import torch
logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _requeset_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _requeset_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            return

        self._requeset_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            self._requeset_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            return

        self._requeset_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            self._requeset_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: Iterable[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. Dict of CPU -> GPU block number.
    blocks_to_swap_in: Dict[int, int]
    # Blocks to swap out. Dict of GPU -> CPU block number.
    blocks_to_swap_out: Dict[int, int]
    # Blocks to copy. Source to a list of dest blocks.
    blocks_to_copy: Dict[int, List[int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    need_score: bool 
    allow_both_swap: bool 

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert self.allow_both_swap or (not (self.blocks_to_swap_in and self.blocks_to_swap_out))
        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()


    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups,
            key=lambda g: (g.seq_group.lora_int_id, g.seq_group.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: Dict[int, int]
    # The blocks to copy.
    blocks_to_copy: Dict[int, List[int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out={},
            blocks_to_copy={},
            num_lookahead_slots=0,
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: Dict[int, int]
    # The blocks to copy.
    blocks_to_copy: Dict[int, List[int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in={},
            blocks_to_copy={},
            num_lookahead_slots=0,
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[SequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        if self.scheduler_config.chunked_prefill_enabled:
            self.prompt_limit = self.scheduler_config.max_model_len
        else:
            self.prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens)

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version="v2" if self.scheduler_config.
            use_v2_block_manager else "v1")

        self.schedule_type = scheduler_config.schedule_type
        self.tbound = -1
        self.starv = -1
        if "starv" in self.schedule_type:
            self.starv = int(self.schedule_type[self.schedule_type.find("starv") +
                                   len("starv"):self.schedule_type.find("period") - 1])
            self.period = int(self.schedule_type[self.schedule_type.find("period") + len("period"):])
            print('Starvation Control: ', self.starv, self.period)
        if "synthetic" in self.schedule_type:
            self.bound = self.schedule_type[self.schedule_type.find("synthetic") + len("synthetic"):]
            print("activate synthetic data bound: ", self.bound)
            self.bound = eval(self.bound)
            #((100,200,100),(-1,-1,200))
            #(-1,-1,100)
            self._finished_req = [0 for _ in self.bound]
        elif "timelimit" in self.schedule_type:
            self.tbound = self.schedule_type[self.schedule_type.find("timelimit") + len("timelimit"):]
            self.tbound = eval(self.tbound)
            self.bound = []
        else:
            self.bound = []
            
        if self.schedule_type.startswith("fcfs") or self.schedule_type == "sjf" or self.schedule_type == 'ljf':
            pass
        elif self.schedule_type.startswith("fifo"):
            self._schedule = self._general_schedule
            self._get_ordered_requests = self._get_fifo_ordered_requests
            self._update_priority = self._update_fifo_priority
            self.need_score = False
        elif self.schedule_type.startswith("srtf"):
            self._schedule = self._general_schedule
            self._get_ordered_requests = self._get_srtf_ordered_requests
            self._update_priority = self._update_srtf_priority
            self.need_score = False
        elif self.schedule_type.startswith("FAKEPO"):
            self._schedule = self._general_schedule
            self._get_ordered_requests = self._get_FAKEPO_ordered_requests
            self._update_priority = self._update_FAKEPO_priority
            self.need_score = False   
        elif self.schedule_type.startswith("PO"):
            self._schedule = self._general_schedule
            self._get_ordered_requests = self._get_PO_ordered_requests
            self._update_priority = self._update_PO_priority
            self.need_score = False
        elif self.schedule_type.startswith("xpt"):
            self.distribution = torch.load(self.schedule_type[self.schedule_type.find("{")+1:self.schedule_type.rfind("}")])
            print("distribution: ", len(self.distribution), self.distribution[0][:100], self.distribution[1][:100])
            self.records = []
            self._schedule = self._general_schedule
            self._get_ordered_requests = self._get_xpt_ordered_requests
            self._update_priority = self._update_xpt_priority
            self.need_score = True
        elif self.schedule_type.startswith("tpt"):
            self._schedule = self._general_schedule
            self._get_ordered_requests = self._get_tpt_ordered_requests
            self._update_priority = self._update_tpt_priority
            self.need_score = True
        elif self.schedule_type.startswith("opt"):
            self._schedule = self._general_schedule
            self._get_ordered_requests = self._get_opt_ordered_requests
            self._update_priority = self._update_opt_priority
            self.need_score = True
        else:
            assert False, f"Not Supported Schedule Type {self.schedule_type}"

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()

        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        self.fake_allocate = scheduler_config.fake_allocate

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        logger.debug(f"add_seq_group {seq_group.request_id}")
        self.waiting.append(seq_group)
        seq_group.idle = 0
        seq_group.runs = 0
        seq_group.pri = 0
        
        #if self.schedule_type.startswith("opt"):
        #    print('[arrival] ', seq_group.request_id)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            running_queue: The queue that contains running requests (i.e.,
                decodes). The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            policy: The sorting policy to sort running_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            A tuple of remaining running queue (should be always 0) after
            scheduling and SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)

        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            # We can have up to 1 running prefill at any given time in running
            # queue, which means we can guarantee chunk size is at least 1.
            assert num_running_tokens != 0
            num_running_seqs = seq_group.get_max_num_running_seqs()

            running_queue.popleft()
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.remove(seq_group.lora_int_id)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    break
            else:
                logger.debug(f"append slot for {seq_group}")
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        # Make sure all queues are updated.
        assert len(running_queue) == 0

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            swapped_queue: The queue that contains swapped out requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            policy: The sorting policy to sort swapped_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining swapped_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                break

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                assert num_new_tokens == 1
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return swapped_queue, SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            if num_new_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_new_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue
            
            if self.fake_allocate:
                can_allocate = AllocStatus.OK
            else:
                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(seq_group)
                
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        f"Input prompt ({num_new_tokens} tokens) is too long"
                        f" and exceeds the capacity of block_manager")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    waiting_queue.popleft()
                    continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()

            if self.fake_allocate:
                self._fake_allocate_and_set_running(seq_group, num_new_tokens)
            else:
                self._allocate_and_set_running(seq_group, num_new_tokens)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        if self.schedule_type == "sjf" and self.waiting:
            self.waiting = deque(sorted(self.waiting, key=lambda req: (req.sampling_params.est_tokens) ))

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            remaining_waiting, prefills = self._schedule_prefills(
                self.waiting, budget, curr_loras, enable_chunking=False)

        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            remaining_running, running_scheduled = self._schedule_running(
                self.running,
                budget,
                curr_loras,
                fcfs_policy,
                enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                remaining_swapped, swapped_in = self._schedule_swapped(
                    self.swapped, budget, curr_loras, fcfs_policy)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=len(prefills.seq_groups),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=merge_dicts(running_scheduled.blocks_to_copy,
                                       swapped_in.blocks_to_copy),
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            need_score=False,
            allow_both_swap=False
        )

    def _get_srtf_ordered_requests(self):
        return sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: req.sampling_params.est_tokens - req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len())

    def _update_srtf_priority(self):
        pass

    def _get_rPO_ordered_requests(self):
        obtain_values = []
        others = []
        for req in  list(self.running) + list(self.swapped):
            if req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len() < 15:
                obtain_values.append(req)
            else:
                others.append(req)

        return sorted(list(self.waiting) + obtain_values, key=lambda req:req.metrics.arrival_time) + sorted(others, key=lambda req: -req.sampling_params.est_tokens)
        
        #return list(self.waiting) + sorted(list(self.running) + list(self.swapped), key=lambda req: req.sampling_params.est_tokens - req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len())

    def _update_rPO_priority(self):
        pass


    def _get_PO_ordered_requests(self):
        obtain_values = []
        others = []
        for req in  list(self.running) + list(self.swapped):
            if req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len() < 15:
                obtain_values.append(req)
            else:
                others.append(req)

        return sorted(list(self.waiting) + obtain_values, key=lambda req:req.metrics.arrival_time) + sorted(others, key=lambda req: req.sampling_params.est_tokens)
        
        #return list(self.waiting) + sorted(list(self.running) + list(self.swapped), key=lambda req: req.sampling_params.est_tokens - req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len())

    def _update_PO_priority(self):
        pass
    def _get_FAKEPO_ordered_requests(self):

        return sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: req.sampling_params.est_tokens)
        
        #return list(self.waiting) + sorted(list(self.running) + list(self.swapped), key=lambda req: req.sampling_params.est_tokens - req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len())

    def _update_FAKEPO_priority(self):
        pass

    def _get_fifo_ordered_requests(self):
        return sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: (req.metrics.arrival_time))

    def _update_fifo_priority(self):
        pass

    def _get_xpt_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)

        if need_aux_scores: 
            self.aux_model.obtain_aux_scores(need_aux_scores)

        all_seqs = list(self.waiting) + list(self.running) + list(self.swapped)
        key, value = self.distribution
        
        for rid in range(len(all_seqs)):
            req = all_seqs[rid]
            if not hasattr(req, "expected_length"):
                score = round(-req.aux_model_score, 2)
                req.expected_length = -10000
                for kid in range(len(key) - 1, -1, -1):
                    if score >= key[kid]:
                        req.expected_length = value[kid] 
                        break
            
        return list(sorted(all_seqs, key=lambda req: req.expected_length - req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len()))

    def _update_xpt_priority(self):
        pass

    def _get_tpt_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)

        if need_aux_scores: 
            self.aux_model.obtain_aux_scores(need_aux_scores)
        
        return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req:(-req.aux_model_score, req.request_id  )))

    def _update_tpt_priority(self):
        pass

    def _get_rtpt_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)

        if need_aux_scores: 
            self.aux_model.obtain_aux_scores(need_aux_scores)
        
        return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: (req.aux_model_score, req.request_id ) ))

    def _update_rtpt_priority(self):
        pass

   
    def _get_opt_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)

        if need_aux_scores: 
            if int(os.environ.get('OPT_TIME', 0)):
                t0 = time.time()
            self.aux_model.obtain_aux_scores(need_aux_scores)
            if int(os.environ.get('OPT_TIME', 0)):
                t1 = time.time()       
                print("OPT-TIME: ", t1 - t0)
        
        if self.starv != -1:
            for r in list(self.waiting) + list(self.running) + list(self.swapped):
                if r.idle >= self.starv:
                    r.pri = -1
                    r.idle = 0
                    r.runs = self.period
                    #print('[promote] ', r.request_id)
                elif r.pri == -1 and r.runs <= 0:
                    r.pri = 0
                    #print('[demote] ', r.request_id)

            #print('[step]')
            ret = list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: (req.pri, -req.aux_model_score )))
        else:
            ret = list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req:  -req.aux_model_score ))

        return ret 

    def _update_opt_priority(self):
        pass

    def _get_ropt_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)

        if need_aux_scores: 
            self.aux_model.obtain_aux_scores(need_aux_scores)
        
        return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: req.aux_model_score ) )

    def _update_ropt_priority(self):
        pass

    def _get_constraint_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)

        if need_aux_scores: 
            ret = self.aux_model.obtain_aux_scores(need_aux_scores)
        
        ranking_scores = [-score for score in ret]
        self.records += ranking_scores
        self.records = sorted(self.records)
        
        return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: -req.aux_model_score))

    def _update_constraint_priority(self):
        pass

    
    def _get_ltr_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)

        if need_aux_scores: 
            self.aux_model.obtain_aux_scores(need_aux_scores)
        
        return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: -req.aux_model_score))


        return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: -req.sampling_params.est_tokens))

        
        wait = list(sorted(list(self.waiting), key=lambda req: -req.sampling_params.est_tokens))
        run = list(sorted(list(self.running) + list(self.swapped), key=lambda req: req.pscore))
        ret = []
        wait_idx = 0
        run_idx = 0
        while wait_idx < len(wait) or run_idx < len(run):
            if wait_idx == len(wait):
                ret.append(run[run_idx])
                run_idx += 1
            elif run_idx == len(run):
                ret.append(wait[wait_idx])
                wait_idx += 1
            else:
                if -wait[wait_idx].sampling_params.est_tokens < -run[run_idx].sampling_params.est_tokens:
                    ret.append(wait[wait_idx])
                    wait_idx += 1
                else:
                    ret.append(run[run_idx])
                    run_idx += 1
        return ret


        if len(self.waiting) > 500: #len(self.running) + len(self.swapped):
            return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: -req.sampling_params.est_tokens))
        else:
            return list(self.waiting) + list(sorted(list(self.running) +
                                               list(self.swapped), key=lambda
                                               req: req.pscore))
        
        ret = list(self.waiting) + list(sorted(list(self.running) +
                                               list(self.swapped), key=lambda
                                               req: req.pscore))
        #return -req.pred_score
        return ret 
        
    def _update_ltr_priority(self):
        '''
        for req in self.running:
#            if not req.pscore and req.pred_score is not None:
            req.pscore = -req.pred_score
        for req in self.swapped:
            #if not req.pscore and req.pred_score is not None:
            req.pscore = -req.pred_score
        '''

    def _general_schedule(self):

        self._update_priority()

        ordered_requests = self._get_ordered_requests()
        original_len = len(self.swapped) + len(self.running) + len(self.waiting)

        #print("budget: ", self.scheduler_config.max_num_batched_tokens, self.scheduler_config.max_num_seqs, len(self.running), len(self.waiting), len(self.swapped))

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        final_budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        enable_chunking = True 
        selected_seq_groups = []
        exe_waiting = []
        exe_swapped_prefill_seq_groups = []
        exe_swapped_decode_seq_groups = []
        exe_running_prefill_seq_groups = []
        exe_running_decode_seq_groups = []
        gpu_block_required = 0
        

        for seq_group in ordered_requests:            
            seq = seq_group.get_seqs()[0]
            if seq_group in remaining_running:
                num_new_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)
                if num_new_tokens == 0:
                    #print(seq_group.get_seqs())
                    assert budget.remaining_token_budget() == 0
                    break

                assert seq_group not in remaining_swapped, f" runs {seq_group}"
                assert seq_group not in remaining_waiting, f" wait {seq_group}"
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_new_tokens == 0
                        or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    break
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)

                seq_group.num_new_tokens = num_new_tokens
                seq_group.num_new_seqs = num_new_seqs

                selected_seq_groups.append(seq_group)
                gpu_block_required += num_new_seqs

                #x1.append(seq_group)

            elif seq_group in remaining_swapped:
                num_new_seqs = seq_group.get_max_num_running_seqs()
                num_new_tokens = self._get_num_new_tokens(seq_group,
                                                        SequenceStatus.SWAPPED,
                                                        enable_chunking, budget)
                num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
                if (num_new_tokens == 0
                        or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    break


                seq_group.num_new_tokens = num_new_tokens
                seq_group.num_new_seqs = num_new_seqs

                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                selected_seq_groups.append(seq_group)
                gpu_block_required += (len(self.block_manager._get_physical_blocks(seq_group)) + num_swapped_seqs)

                
            elif seq_group in remaining_waiting:
                waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
                num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
                if num_new_tokens > self.prompt_limit:
                    assert False, "req exceed prompt limit"
                #can allocate later
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_new_tokens == 0
                        or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    #print('not appending')
                    break

                seq_group.num_new_tokens = num_new_tokens
                seq_group.num_new_seqs = num_new_seqs
                #print("append seq group: ", seq_group)
                selected_seq_groups.append(seq_group)
                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                gpu_block_required += len(seq.logical_token_blocks)
            else:
                
                assert False, "seqgroup not in all lists"

        for seq_group in selected_seq_groups:
            ordered_requests.remove(seq_group)
        
        #print("before remain: ", len(selected_seq_groups), len(ordered_requests), budget.num_curr_seqs)

        _, execute_pinned_requests, preempted, swapped_out, blocks_to_swap_out, blocks_to_swap_in = self.reserve_free_blocks(gpu_block_required, selected_seq_groups, ordered_requests, remaining_running, final_budget)
        blocks_to_copy = {}

        for seq_group in execute_pinned_requests:
            if seq_group in remaining_waiting:
                remaining_waiting.remove(seq_group)
                if self.block_manager.can_allocate(seq_group) == AllocStatus.OK:
                    self._allocate_and_set_running(seq_group, seq_group.num_new_tokens)
                    exe_waiting.append(ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=seq_group.num_new_tokens))
                    
                    #final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                    #final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)
                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs
                else:
                    assert False, "can not append new req"
            elif seq_group in remaining_running:
                remaining_running.remove(seq_group)
                if self.block_manager.can_append_slots(seq_group):
                    self._append_slots(seq_group, blocks_to_copy)

                    is_prefill = seq_group.is_prefill()
                    #print("prefill run: ", is_prefill)
                    if is_prefill:
                        exe_running_prefill_seq_groups.append(
                            ScheduledSequenceGroup(
                                seq_group=seq_group,
                                token_chunk_size=seq_group.num_new_tokens))
                    else:
                        exe_running_decode_seq_groups.append(
                            ScheduledSequenceGroup(seq_group=seq_group,
                                                token_chunk_size=1))

                    #final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                    #final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)
                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs

                else:
                    assert False

            elif seq_group in remaining_swapped:
                remaining_swapped.remove(seq_group)
                if self.block_manager.can_append_slots(seq_group):
                    self._append_slots(seq_group, blocks_to_copy)

                    is_prefill = seq_group.is_prefill()
                    #print("swapped: ", seq_group, is_prefill)
                    if is_prefill:
                        exe_swapped_prefill_seq_groups.append(
                            ScheduledSequenceGroup(seq_group,
                                                token_chunk_size=seq_group.num_new_tokens))
                    else:
                        assert seq_group.num_new_tokens == 1
                        exe_swapped_decode_seq_groups.append(
                            ScheduledSequenceGroup(seq_group, token_chunk_size=1))

                    #final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                    #final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)
                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs

                else:
                    assert False
            else:
                assert False 
        #print("prefill decode: ", len(exe_running_decode_seq_groups), len(exe_running_prefill_seq_groups), len(remaining_running))
        #assert len(remaining_running) == 0
        prefills = SchedulerPrefillOutputs(
            seq_groups=exe_waiting,
            ignored_seq_groups=[],
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))
        swapped_in = SchedulerSwappedInOutputs(
            decode_seq_groups=exe_swapped_decode_seq_groups,
            prefill_seq_groups=exe_swapped_prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))
        running_scheduled = SchedulerRunningOutputs(
            decode_seq_groups=exe_running_decode_seq_groups,
            prefill_seq_groups=exe_running_prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

        assert (final_budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs, f" num req: {budget.num_curr_seqs} {self.scheduler_config.max_num_seqs}"

        #print("extend: ", len(remaining_running), len(prefills.seq_groups), len(running_scheduled.decode_seq_groups), len(running_scheduled.prefill_seq_groups), len(running_scheduled.prefill_seq_groups), len(swapped_in.decode_seq_groups), len(swapped_in.prefill_seq_groups))
        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        #print("tt: ", list([x.seq_group.is_prefill() for x in prefills.seq_groups]), list([x.seq_group.is_prefill() for x in running_scheduled.prefill_seq_groups]), list([x.seq_group.is_prefill() for x in running_scheduled.decode_seq_groups]))
        #xx = list(prefills.seq_groups +
        #                          running_scheduled.prefill_seq_groups +
        #                          swapped_in.prefill_seq_groups +
        #                          running_scheduled.decode_seq_groups +
        #                          swapped_in.decode_seq_groups)
        #print("jj: ", [x.seq_group.is_prefill() for x in xx], list([x.seq_group.is_prefill() for x in prefills.seq_groups]), list([x.seq_group.is_prefill() for x in running_scheduled.decode_seq_groups]), list([x.seq_group.is_prefill() for x in prefills.seq_groups + running_scheduled.decode_seq_groups]))
        all_pri = list(self.swapped) + list(self.running) + list(self.waiting)
        assert len(self.swapped) + len(self.running) + len(self.waiting) == original_len
        ret = SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=final_budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=merge_dicts(running_scheduled.blocks_to_copy,
                                       swapped_in.blocks_to_copy),
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            need_score=self.need_score,
            allow_both_swap=True
        )
        running_this_step = [r.seq_group for r in ret.scheduled_seq_groups]
        for seq in all_pri:
            if seq in running_this_step:
                if seq.pri == -1:
                    seq.runs -= 1
                seq.idle = 0
            else:
                seq.idle += 1
        #if self.schedule_type.startswith("opt"):
        #    print('[running] ', [r.request_id for r in running_this_step])
        #print('----------')
        #print('running: ', [r.request_id for r in running_this_step])
        #print('all: ', [r.request_id for r in all_pri])
        #print('----------')
        
        return ret


    def reserve_free_blocks(self, num_blocks_needed, pinned_requests: List[SequenceGroup], priority_requests, remaining_running, final_budget):

        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_swap_in: Dict[int, int] = {}
        
        preempted = []
        swapped_out = []

        num_swap_out_blocks_needed = (
            num_blocks_needed
            - self.block_manager.gpu_allocator.get_num_free_blocks() \
            + self.block_manager.watermark_blocks
        )
        swap_out_needed = num_swap_out_blocks_needed > 0

        # the pinned requests we really execute
        execute_pinned_requests = pinned_requests.copy()
        # the pinned requests we put back due to swapped out
        swapped_pinned_requests: List[SequenceGroup] = []

        # swap out low priority requests if GPU blocks are not enough
        if swap_out_needed:
            pinned_request_ids = set(
                [request.request_id for request in pinned_requests]
            )
            # swap out from the lowest priority request
            for request in reversed(priority_requests): 
                # pinned request must have already been popped from MLFQ,
                assert request.request_id not in pinned_request_ids
                if num_swap_out_blocks_needed <= 0:
                    break
                if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                    num_swap_out_blocks_needed -= len(self.block_manager._get_physical_blocks(request))
                    #print("preempt1: ", request, request.is_prefill(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_output_len(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_prompt_len(),  request.seqs_dict[next(iter(request.seqs_dict))].data.get_num_computed_tokens())
                    preempted_mode = self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                    #print("preempt1 done: ", request, request.is_prefill(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_output_len(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_prompt_len(),  request.seqs_dict[next(iter(request.seqs_dict))].data.get_num_computed_tokens())
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(request)
                    else:
                        swapped_out.append(request)

                    #print("preempt: ", request in remaining_running, request in execute_pinned_requests)
                    if request in remaining_running:
                        assert request not in execute_pinned_requests
                        remaining_running.remove(request)
                    else:
                        execute_pinned_requests.remove(request)

            if num_swap_out_blocks_needed > 0:
                # if we still need to swap out blocks, swap out pinned requests
                # location of pinned requests may be in CPU/GPU or none now
                while num_swap_out_blocks_needed > 0 and len(execute_pinned_requests) > 0:
                    request = execute_pinned_requests.pop(-1)
                    swapped_pinned_requests.append(request)
                    if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                        num_swap_out_blocks_needed -= request.num_seqs(status=SequenceStatus.RUNNING)
                        num_swap_out_blocks_needed -= len(self.block_manager._get_physical_blocks(request))
                        #print("preempt2: ", request, request.is_prefill(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_output_len(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_prompt_len(),  request.seqs_dict[next(iter(request.seqs_dict))].data.get_num_computed_tokens())
                        preempted_mode = self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                        #print("preempt2 done: ", request, request.is_prefill(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_output_len(), request.seqs_dict[next(iter(request.seqs_dict))].data.get_prompt_len(),  request.seqs_dict[next(iter(request.seqs_dict))].data.get_num_computed_tokens())
                        #print("preempt: ", request in remaining_running, preempted_mode)
                        
                        remaining_running.remove(request)
                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.append(request)
                        else:
                            swapped_out.append(request)

                    elif (len(request.get_seqs(status=SequenceStatus.SWAPPED))):

                        num_swap_out_blocks_needed -= (len(self.block_manager._get_physical_blocks(request)) + request.num_seqs(status=SequenceStatus.SWAPPED))
                    else:
                        num_swap_out_blocks_needed -= len(request.get_seqs()[0].logical_token_blocks)  
                        

            # swap block is required by waiting request and we already put it back
            assert num_swap_out_blocks_needed <= 0

        # swap in pinned requests if needed
        for seq_group in execute_pinned_requests:
            if (len(seq_group.get_seqs(status=SequenceStatus.SWAPPED))):
                self._swap_in(seq_group, blocks_to_swap_in)

            final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
            final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)
            #del seq_group.num_new_tokens
            #del seq_group.num_new_seqs

        # swap in high priority requests if (1) no swap out gets executed, avoid ping-pong swapping (2) proactive swapping is enabled
        # this is ok
        if not swap_out_needed:
            #swap_quata = self.scheduler_config.max_num_seqs
            for request in priority_requests:                 
            #    if swap_quata <= 0:
            #        break

                if (len(request.get_seqs(status=SequenceStatus.SWAPPED))):

                    num_new_seqs = request.get_max_num_running_seqs()
                    num_new_tokens = self._get_num_new_tokens(request,
                                                        SequenceStatus.SWAPPED,
                                                        enable_chunking=True, budget=final_budget)


                    # swap in the request if there are enough free blocks
                    if (
                        self.block_manager.can_swap_in(request)
                    ) and (num_swap_out_blocks_needed + len(self.block_manager._get_physical_blocks(request)) + request.num_seqs(status=SequenceStatus.SWAPPED)) < 0 \
                        and (num_new_tokens > 0 and final_budget.can_schedule(num_new_tokens=num_new_tokens, num_new_seqs=num_new_seqs)):
                        
                        #print("Xswap in : ", request, request.is_prefill(), request.get_max_num_running_seqs(), sum(seq.get_num_new_tokens() for seq in request.get_seqs(status=SequenceStatus.SWAPPED)))
                        request.num_new_seqs = request.get_max_num_running_seqs()
                        request.num_new_tokens = sum(seq.get_num_new_tokens() for seq in request.get_seqs(status=SequenceStatus.SWAPPED))
                        self._swap_in(request, blocks_to_swap_in)

                        final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                        final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)

                        execute_pinned_requests.append(request)
                        num_swap_out_blocks_needed += (len(self.block_manager._get_physical_blocks(request)) + request.num_seqs(status=SequenceStatus.SWAPPED))
                    else:
                        break
                # reduce the quata no matter if the request needs swapping in
                #swap_quata -= 1


        return swapped_pinned_requests, execute_pinned_requests, preempted, swapped_out, blocks_to_swap_out, blocks_to_swap_in


    def _schedule_chunked_prefill(self):
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # Decoding should be always scheduled first by fcfs.
        fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            fcfs_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, fcfs_policy)
        #print("prefill")
        # Schedule new prefills.
        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs, f" num req: {budget.num_curr_seqs} {self.scheduler_config.max_num_seqs}"

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=merge_dicts(running_scheduled.blocks_to_copy,
                                       swapped_in.blocks_to_copy),
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            need_score=False,
            allow_both_swap=False
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # Appending slots only occurs in decoding.
        is_prefill = False

        return self.block_manager.can_append_slots(
            seq_group=seq_group,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def _can_swap_in(self, seq_group: SequenceGroup) -> bool:
        # Swapping in is considered decode.
        is_prefill = False

        return self.block_manager.can_swap_in(
            seq_group=seq_group,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        #print("xx: ", [x.seq_group.is_prefill() for x in scheduler_outputs.scheduled_seq_groups])
        now = time.time()
        
        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)
            #print("xp: ", seq_group.is_prefill())
            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                if self.fake_allocate:
                    block_tables[seq_id] = self.block_manager.get_fake_block_table_and_delete(seq)
                else:
                    block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))
            
            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = seq_group.is_prefill()
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.num_prefill_groups > 0 else None,
                need_score=scheduler_outputs.need_score
            )
            seq_group_metadata_list.append(seq_group_metadata)
        #print(prefills)
        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        for req in self.running:
            if req.is_finished():
                #if self.schedule_type.startswith("opt"):
                #    print('[finished] ', req.request_id)
                count_token = req.seqs_dict[next(iter(req.seqs_dict))].data.get_output_len()
                for ib, bound in enumerate(self.bound):
                    if (bound[0] == -1 or count_token >= bound[0]) and (bound[1] == -1 or count_token <= bound[1]) :
                        #print("finished: ", self._finished_req)
                        self._finished_req[ib] += 1
                        break
        all_finished = True
        for idx in range(len(self.bound)):
            cnt, ib = self._finished_req[idx], self.bound[idx]
            #print(cnt, ib, self.bound, self._finished_req, idx)
            if cnt < ib[2]:
                all_finished = False
        if self.tbound != -1 and time.time() - self.start_time >= self.tbound:
            self.running = deque([])
            self.swapped = deque([])
            self.waiting = deque([])
        if all_finished and len(self.bound) > 0:
            self.running = deque([])
            self.swapped = deque([])
            self.waiting = deque([])
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())


    def _allocate_and_set_running(self, seq_group: SequenceGroup,
                                  num_new_tokens: int) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _fake_allocate_and_set_running(self, seq_group: SequenceGroup,
                                  num_new_tokens: int) -> None:
        self.block_manager.fake_allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (Dict[int, List[int]]): A dictionary mapping source
                block indices to lists of destination block indices. This
                dictionary is updated with the new source and destination block
                indices for the appended slots.
        """
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)

            for src, dests in cows.items():
                if src not in blocks_to_copy:
                    blocks_to_copy[src] = []
                blocks_to_copy[src].extend(dests)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        seq_group.count_swap_out()

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.
        """
        if is_prefill:
            return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, seq_group: SequenceGroup,
                            status: SequenceStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        """Get the next new tokens to compute for a given sequence group
            that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.
        """
        num_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            num_new_tokens += seq.get_num_new_tokens()
        # Chunk if a running request cannot fit in.
        # If number of seq > 1, it means it is doing beam search in a
        # decode phase. Do not chunk in that case.
        if enable_chunking and len(seqs) == 1:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens
