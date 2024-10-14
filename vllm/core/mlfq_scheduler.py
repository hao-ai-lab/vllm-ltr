import enum
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
logger = init_logger(__name__)


from vllm.core.scheduler import PreemptionMode, SchedulingBudget, SchedulerOutputs, ScheduledSequenceGroup, SchedulerRunningOutputs, SchedulerSwappedInOutputs, SchedulerPrefillOutputs

class MLFQScheduler:
    class Priority_Queue:
        def __init__(self, priority: int):
            self.priority = priority
            self.requests = []

        def push_front(self, request) -> None:
            self.requests.insert(0, request)

        def push_back(self, request) -> None:
            self.requests.append(request)

        def pop_front(self):
            return self.requests.pop(0)
        
        def extend_front(self, requests_deque: deque) -> None:
            for request in reversed(requests_deque):
                self.push_front(request)
        
        def print_queue(self):
            print(f"Priority {self.priority}: ", end='')
            print(", ".join([str(request.request_id) for request in self.requests]))

        def __len__(self):
            return len(self.requests)

    class Priority_Queues:
        def __init__(self):
            self.queues: List[MLFQScheduler.Priority_Queue] = []

        def add_new_queue(self, priority: int) -> None:
            if priority >= len(self.queues):
                for p in range(len(self.queues), priority + 1):
                    self.queues.append(MLFQScheduler.Priority_Queue(p))

        def pop_front(self) -> None:
            for priority in range(len(self.queues)):
                if len(self.queues[priority]) > 0:
                    return self.queues[priority].pop_front()

        def push_back(self, request) -> None:
            self.add_new_queue(request.get_priority())
            self.queues[request.get_priority()].push_back(request)

        def push_front(self, request) -> None:
            self.add_new_queue(request.get_priority())
            self.queues[request.get_priority()].push_front(request)

        def del_request(self, request_id: int) -> None:
            for queue in self.queues:
                for i, request in enumerate(queue.requests):
                    if request.request_id == request_id:
                        del queue.requests[i]
                        return

        def queue_list(self) -> List:
            ret = []
            for queue in self.queues:
                for req in queue.requests:
                    ret.append(req)
            return ret

        def get_num_requests_in_top_queue(self, num_queues=2) -> int:
            for priority in range(len(self.queues)):
                if len(self.queues[priority]) > 0:
                    num_requests_in_top_queue = 0
                    for i in range(num_queues):
                        if priority + i < len(self.queues):
                            num_requests_in_top_queue += len(self.queues[priority + i])
                    return num_requests_in_top_queue
            return 0
        
        def extend_front(self, requests_deque: deque) -> None:
            for request in requests_deque:
                self.add_new_queue(request.get_priority())
                self.queues[request.get_priority()].extend_front(deque([request]))
        
        def print_queues(self):
            if not self.queues:
                print("No requests in the queues.")
                return
            for queue in self.queues:
                if len(queue) > 0:
                    queue.print_queue()
                else:
                    print(f"Priority {queue.priority}: No requests")

        def __len__(self):
            return sum([len(q) for q in self.queues])

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
        
        self.iteration_num = 0
        self.priority_queues = self.Priority_Queues() 
        self.swapped = [] 
        # they refer to, request with block on CPU,  request block on GPU and will execute this step, request block on GPU but not execute in this step
       
        self.schedule_type = scheduler_config.schedule_type
        self.enable_starvation_prevent = scheduler_config.enable_starvation_prevent

        def find_config(s, c):
            st = s[s.find(c) + len(c):]
            if '-' in st:
                return st[:st.find('-')  ]
            return st
        # Multi-level Feedback Queue
        # Since pipeline parallelism is used, there may be multiple batches under processing.
        # Just some magic numbers, need to be tuned.
        #example: 
        #schedule_type = mlfq-async-quant0.01-thres2-starv3.0-starp100
        if 'quant' in self.schedule_type:
            self.base_quantum = float(find_config(self.schedule_type, 'quant'))
        else:
            self.base_quantum = 0.01  # 10 ms

        if 'thres' in self.schedule_type:
            self.threshold = float(find_config(self.schedule_type, 'thres'))
        else:
            self.threshold = 2

        if 'starv' in self.schedule_type:
            self.starvation_threshold = float(find_config(self.schedule_type, 'starv'))
        else:
            self.starvation_threshold = 3.  # 3 seconds

        if 'starp' in self.schedule_type:
            self.starvation_period = float(find_config(self.schedule_type, 'starp'))
        else:
            self.starvation_period = 100  # 1000 iterations
            
        print("MLFQ Tune info Log")
        print(f"Base Quant {self.base_quantum}")
        print(f"Threshold for Priority  Queue {self.threshold}")
        if self.enable_starvation_prevent:
            print(f"Starvation Prevent Enabled")
            print(f"Starvation Threshold: {self.starvation_threshold}")
            print(f"Starvation Period: {self.starvation_period}")
        else:
            print(f"Starvaition Disabled")
        

        if self.scheduler_config.chunked_prefill_enabled:
            self.prompt_limit = self.scheduler_config.max_model_len
        else:
            self.prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version="v2" if self.scheduler_config.
            use_v2_block_manager else "v1")

        # Instantiate the scheduling policy.
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        
    
    def print_batch_queues(self):
        print(f"\nIteration Counter {self.iteration_num}")
        print(f"Scheduled Request for this step: ", end='')
        print(", ".join([str(request.request_id) for request in self.batch_queues]))


    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        logger.debug(f"add_seq_group {seq_group.request_id}")
        seq_group.set_priority(0)
        self.priority_queues.push_back(seq_group)
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        #print("ABORTING=======================")
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
        for request_id in request_ids:
            self.priority_queues.del_request(request_id)
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
                    
        aborted_groups: List[SequenceGroup] = []
        for seq_group in self.real_running:
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
            self.real_running.remove(aborted_group)
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

    def has_unfinished_seqs_old(self) -> bool:
        return self.batch_queues or len(self.priority_queues) > 0

    def get_num_unfinished_seq_groups_old(self) -> int:
        return len(self.priority_queues)
    
    def reserve_free_blocks_old(self, num_blocks_needed, pinned_requests: List[SequenceGroup]):
        blocks_to_swap_out = {}
        blocks_to_swap_in = {}
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
            for priority in reversed(range(len(self.priority_queues.queues))):
                for request in reversed(
                    self.priority_queues.queues[priority].requests
                ):
                    # pinned request must have already been popped from MLFQ,
                    assert request.request_id not in pinned_request_ids
                    if num_swap_out_blocks_needed <= 0:
                        break
                    if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                        num_swap_out_blocks_needed -= self.block_manager.get_physical_blocks_num(request)
                        self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                        if request in self.running_but_not_finished:
                            assert request not in execute_pinned_requests
                            self.running_but_not_finished.remove(request)
                        else:
                            execute_pinned_requests.remove(request)
                if num_swap_out_blocks_needed <= 0:
                    break
            if num_swap_out_blocks_needed > 0:
                # if we still need to swap out blocks, swap out pinned requests
                # location of pinned requests may be in CPU/GPU or none now
                while num_swap_out_blocks_needed > 0 and len(execute_pinned_requests) > 0:
                    request = execute_pinned_requests.pop(-1)
                    swapped_pinned_requests.append(request)
                    if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                        num_swap_out_blocks_needed -= request.num_seqs(status=SequenceStatus.RUNNING)
                        num_swap_out_blocks_needed -= self.block_manager.get_physical_blocks_num(request)
                        self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                    elif (len(request.get_seqs(status=SequenceStatus.SWAPPED))):
                        self.swapped.append(request)
                        num_swap_out_blocks_needed -= self.block_manager.get_physical_blocks_num(request) + request.num_seqs(status=SequenceStatus.SWAPPED)
                    else:
                        num_swap_out_blocks_needed -= len(request.get_seqs()[0].logical_token_blocks)  
                        

            # swap block is required by waiting request and we already put it back
            assert num_swap_out_blocks_needed <= 0

        #first swap out, then swap in
        
        count_swap_in = 0 
        # swap in pinned requests if needed
        for request in execute_pinned_requests:
            if (len(request.get_seqs(status=SequenceStatus.SWAPPED))):
                self._swap_in(request, blocks_to_swap_in)
                count_swap_in += 1
        
        # swap in high priority requests if (1) no swap out gets executed, avoid ping-pong swapping (2) proactive swapping is enabled
        # this is ok
        if not swap_out_needed:
            swap_quata = self.scheduler_config.max_num_seqs
            for priority in range(len(self.priority_queues.queues)):
                if swap_quata <= 0:
                    break
                for request in self.priority_queues.queues[priority].requests:
                    if (len(request.get_seqs(status=SequenceStatus.SWAPPED))):
                        # swap in the request if there are enough free blocks
                        if (
                            self.block_manager.can_swap_in(request)
                        ) and (num_swap_out_blocks_needed + self.block_manager.get_physical_blocks_num(request) + request.num_seqs(status=SequenceStatus.SWAPPED)) < 0:
                           self._swap_in(request, blocks_to_swap_in)
                           execute_pinned_requests.append(request)
                           request.type = 'decode'
                           count_swap_in += 1
                           self.priority_queues.del_request(request.request_id)
                           num_swap_out_blocks_needed += (self.block_manager.get_physical_blocks_num(request) + request.num_seqs(status=SequenceStatus.SWAPPED))
                           self.swapped.remove(request)
                        else:
                            break
                    # reduce the quata no matter if the request needs swapping in
                    swap_quata -= 1
                    if swap_quata <= 0:
                        break
        #print('count swap in: ', count_swap_in)

        return swapped_pinned_requests, execute_pinned_requests, blocks_to_swap_out, blocks_to_swap_in
        
    def prevent_starvation(self) -> None:
        """
        Prevent starvation of the request by promoting it to the top queue.
        """
        promote_reqs = []
        # print(f"Starvation prevent triggered for iteration {self.iteration_num}")
        cur_time = time.time()
        
        for q in self.priority_queues.queues:
            buffer = []
            while len(q) > 0:
                request = q.pop_front()
                if cur_time - request.lst_process_time >= self.starvation_threshold:
                    promote_reqs.append(request)
                else:
                    buffer.append(request)
            
            for request in buffer:
                q.push_back(request)
        
        # promote the requests in starvation
        for request in promote_reqs:
            request.set_priority(0)
            self.priority_queues.push_front(request)

    def reserve_free_blocks(self, num_blocks_needed, pinned_requests: List[SequenceGroup], remaining_running, final_budget):

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
            for priority in reversed(range(len(self.priority_queues.queues))):
                for request in reversed(
                    self.priority_queues.queues[priority].requests
                ):

                    # pinned request must have already been popped from MLFQ,
                    assert request.request_id not in pinned_request_ids
                    if num_swap_out_blocks_needed <= 0:
                        break
                    if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                        num_swap_out_blocks_needed -= len(self.block_manager._get_physical_blocks(request))
                        preempted_mode = self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.append(request)
                        else:
                            swapped_out.append(request)

                        if request in remaining_running:
                            assert request not in execute_pinned_requests
                            remaining_running.remove(request)
                        else:
                            execute_pinned_requests.remove(request)
                if num_swap_out_blocks_needed <= 0:
                    break

            if num_swap_out_blocks_needed > 0:
                # if we still need to swap out blocks, swap out pinned requests
                # location of pinned requests may be in CPU/GPU or none now
                while num_swap_out_blocks_needed > 0 and len(execute_pinned_requests) > 0:
                    request = execute_pinned_requests.pop(-1)
                    swapped_pinned_requests.append(request)
                    if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                        num_swap_out_blocks_needed -= request.num_seqs(status=SequenceStatus.RUNNING)
                        num_swap_out_blocks_needed -= len(self.block_manager._get_physical_blocks(request))
                        preempted_mode = self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                        
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

        # swap in high priority requests if (1) no swap out gets executed, avoid ping-pong swapping (2) proactive swapping is enabled
        # this is ok
        if not swap_out_needed:
            #swap_quata = self.scheduler_config.max_num_seqs
            for priority in range(len(self.priority_queues.queues)):               
                for request in self.priority_queues.queues[priority].requests:
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
                            
                            request.num_new_seqs = request.get_max_num_running_seqs()
                            request.num_new_tokens = sum(seq.get_num_new_tokens() for seq in request.get_seqs(status=SequenceStatus.SWAPPED))
                            self._swap_in(request, blocks_to_swap_in)

                            final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                            final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)

                            execute_pinned_requests.append(request)
                            num_swap_out_blocks_needed += (len(self.block_manager._get_physical_blocks(request)) + request.num_seqs(status=SequenceStatus.SWAPPED))
                        else:
                            break


        return swapped_pinned_requests, execute_pinned_requests, preempted, swapped_out, blocks_to_swap_out, blocks_to_swap_in


    def _mlfq_schedule(self):

        ordered_requests = self.priority_queues.queue_list()

        #print(f" start {len(self.running)} {len(self.swapped)} {len(self.waiting)} {self.get_num_unfinished_seq_groups_old()} ")

        original_len = self.get_num_unfinished_seq_groups()
        assert self.get_num_unfinished_seq_groups() == self.get_num_unfinished_seq_groups_old(), f" count {len(self.running)} {len(self.swapped)} {len(self.waiting)} {self.get_num_unfinished_seq_groups_old()} "

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
        
        #ordered_requests includes all requests
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
                    break

                seq_group.num_new_tokens = num_new_tokens
                seq_group.num_new_seqs = num_new_seqs
                selected_seq_groups.append(seq_group)
                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                gpu_block_required += len(seq.logical_token_blocks)
            else:
                
                assert False, "seqgroup not in all lists"

        for seq_group in selected_seq_groups:
            self.priority_queues.del_request(seq_group.request_id)
        
        #print(f" after del count {len(self.running)} {len(self.swapped)} {len(self.waiting)} {self.get_num_unfinished_seq_groups_old()} ")
        swapped_pinned_requests, execute_pinned_requests, preempted, swapped_out, blocks_to_swap_out, blocks_to_swap_in = self.reserve_free_blocks(gpu_block_required, selected_seq_groups, remaining_running, final_budget)
        blocks_to_copy = {}
        #print(f" after reserve count {len(self.running)} {len(self.swapped)} {len(self.waiting)} {self.get_num_unfinished_seq_groups_old()} ")

        for seq_group in reversed(swapped_pinned_requests):
            self.priority_queues.push_front(seq_group)

        for seq_group in execute_pinned_requests:
            if seq_group in remaining_waiting:
                remaining_waiting.remove(seq_group)
                if self.block_manager.can_allocate(seq_group) == AllocStatus.OK:
                    self._allocate_and_set_running(seq_group, seq_group.num_new_tokens)
                    seq_group.lst_process_time = time.time()
                    exe_waiting.append(ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=seq_group.num_new_tokens))
                    
                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs
                else:
                    assert False, "can not append new req"
            elif seq_group in remaining_running:
                remaining_running.remove(seq_group)
                if self.block_manager.can_append_slots(seq_group):
                    self._append_slots(seq_group, blocks_to_copy)
                    seq_group.lst_process_time = time.time()
                    is_prefill = seq_group.is_prefill()
                    if is_prefill:
                        exe_running_prefill_seq_groups.append(
                            ScheduledSequenceGroup(
                                seq_group=seq_group,
                                token_chunk_size=seq_group.num_new_tokens))
                    else:
                        exe_running_decode_seq_groups.append(
                            ScheduledSequenceGroup(seq_group=seq_group,
                                                token_chunk_size=1))

                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs

                else:
                    assert False

            elif seq_group in remaining_swapped:
                remaining_swapped.remove(seq_group)
                if self.block_manager.can_append_slots(seq_group):
                    self._append_slots(seq_group, blocks_to_copy)
                    seq_group.lst_process_time = time.time()
                    is_prefill = seq_group.is_prefill()
                    if is_prefill:
                        exe_swapped_prefill_seq_groups.append(
                            ScheduledSequenceGroup(seq_group,
                                                token_chunk_size=seq_group.num_new_tokens))
                    else:
                        assert seq_group.num_new_tokens == 1
                        exe_swapped_decode_seq_groups.append(
                            ScheduledSequenceGroup(seq_group, token_chunk_size=1))

                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs

                else:
                    assert False
            else:
                assert False 
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
            need_score=False,
            allow_both_swap=True
        )
        self.real_running = [r.seq_group for r in ret.scheduled_seq_groups]

        #print(f" after maintain count {len(self.running)} {len(self.swapped)} {len(self.waiting)} {self.get_num_unfinished_seq_groups_old()} ")

        assert self.get_num_unfinished_seq_groups_old() == len(self.waiting) + len(self.swapped) + len(self.running) - len(self.real_running), f" count {len(self.running)} {len(self.swapped)} {len(self.waiting)} {len(self.real_running)} {self.get_num_unfinished_seq_groups_old()} "
        assert self.get_num_unfinished_seq_groups() == original_len

        return ret

    def _schedule(self) -> SchedulerOutputs:

        return self._mlfq_schedule()

        blocks_to_swap_in = {}
        blocks_to_swap_out = {}
        blocks_to_copy = {}
        
            

        running = [] # Temp array we use to store the request we wish to run in this step.
        num_curr_seqs = 0
        num_batched_tokens = 0
        seq_lens: List[int] = []

        queue_list = self.priority_queues.queue_list()
        gpu_block_required = 0 # extra gpu block we need to place all request in running to GPU
        assert  self.batch_queues == []
        for seq_group in queue_list:
            seq_group.type = ""

        cnt_run = 0
        cnt_swap = 0
        cnt_prefill = 0
        for seq_group in queue_list:
            seq_group.type = ""

        for seq_group in queue_list:            
            seq = seq_group.get_seqs()[0]
            
            if seq_group in self.running_but_not_finished:
                num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
                new_seq_lens = seq_lens + [seq.get_len() for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING)]
                num_batched_tokens += seq_group.num_unfinished_seqs() * self.num_decoding_tokens_per_seq 
                if num_curr_seqs + num_seqs <= self.scheduler_config.max_num_seqs \
                        and num_batched_tokens <= self.scheduler_config.max_num_batched_tokens:
                    seq_group.type = "decode"
                    cnt_run += 1
                    num_curr_seqs += num_seqs
                    running.append(seq_group)  #TODO 

                    #print('block: ', seq_group.is_finished(),
                    #      seq_group.num_seqs(status=SequenceStatus.RUNNING),
                    #      seq_group.num_seqs(status=SequenceStatus.SWAPPED),
                    #      seq_group.num_seqs(status=SequenceStatus.WAITING),
                    #      seq_group.get_seqs()[0].status)
                    #print("seq: ", seq_group.get_seqs()[0].status)
                    # gpu block required for append slot
                    gpu_block_required += seq_group.num_seqs(status=SequenceStatus.RUNNING)
                    seq_lens = new_seq_lens
                    self.running_but_not_finished.remove(seq_group)
                else:
                    # seq_number exceed or seq_len exceed, abort the loop
                    break
            
            if seq_group in self.swapped:
                num_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
                new_seq_lens = seq_lens + [seq.get_len() for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED)]
                num_batched_tokens += seq_group.num_unfinished_seqs() * self.num_decoding_tokens_per_seq 
                if num_curr_seqs + num_seqs <= self.scheduler_config.max_num_seqs \
                        and num_batched_tokens <= self.scheduler_config.max_num_batched_tokens:
                    seq_group.type = "decode"
                    cnt_swap += 1
                    num_curr_seqs += num_seqs
                    num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
                    running.append(seq_group)
                    gpu_block_required += (len(self.block_manager._get_physical_blocks(seq_group)) + num_swapped_seqs)
                    self.swapped.remove(seq_group)

                    #print('swapx: ', seq_group.is_finished(),
                    #      seq_group.num_seqs(status=SequenceStatus.RUNNING),
                    #      seq_group.num_seqs(status=SequenceStatus.SWAPPED),
                    #      seq_group.num_seqs(status=SequenceStatus.WAITING),
                    #      seq_group.get_seqs()[0].status)
                    #print("seq: ", seq_group.get_seqs()[0].status)
              
                    # assert num_seqs > 0
                    # if self.block_manager.can_swap_in(seq_group): #TODO GPU contains request with lower priority queue
                    #     self._swap_in(seq_group, blocks_to_swap_in)
                    #     self.swapped.remove(seq_group)
                    #     running.append(seq_group)
                    #     seq_lens = new_seq_lens                  
                    # else:
                    #     available = False
                else:
                    break


            
            # as these request always get the highest prioirty. we don't make instant swap in here. 
            if seq_group not in self.swapped and seq_group not in self.running_but_not_finished and seq_group not in running:
                num_required_blocks = len(seq.logical_token_blocks)       
                num_new_seqs = seq_group.get_max_num_running_seqs()
                num_prompt_tokens = seq.get_len()
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens += seq_group.get_num_unprefilled()
        
                if num_curr_seqs + num_new_seqs <= self.scheduler_config.max_num_seqs \
                        and num_batched_tokens <= self.scheduler_config.max_num_batched_tokens:
                    seq_group.type = "prefill"
                    cnt_prefill += 1
                    num_curr_seqs += num_new_seqs
                    running.append(seq_group)
                    gpu_block_required += num_required_blocks
                    if num_prompt_tokens > self.prompt_limit:
                        assert False, "PROMPT TOKENS EXCEED LIMIT"
                    seq_lens = new_seq_lens
                else:
                    break
                    
        for seq_group in running:
            assert seq_group.type != "", "ADD An EMPTY REQUEST"
            self.priority_queues.del_request(seq_group.request_id)
            if seq_group in self.swapped or seq_group in self.running_but_not_finished:
                assert False
                      
        swapped_pinned_requests, execute_pinned_requests, blocks_to_swap_out, blocks_to_swap_in = self.reserve_free_blocks(gpu_block_required, running)

        running = execute_pinned_requests
        
        # we already swap these request in function self.reserve_free_blocks
        # assert swapped_pinned_requests == []
        for seq_group in reversed(swapped_pinned_requests):
            self.priority_queues.push_front(seq_group)
            
        self.iteration_num += 1
        
        if self.enable_starvation_prevent:
            if self.iteration_num % self.starvation_period == 0:
                self.prevent_starvation()

        temp_num_batched_token = 0
        running_with_prefill_start = []
        num_prompt = 0
        
        # place prefill in the front
        for seq_group in running:
            if seq_group.type == "prefill": #prefill
                if self.block_manager.can_allocate(seq_group) == AllocStatus.OK:
                    seq_group.lst_process_time = time.time()
                    running_with_prefill_start.append(seq_group)
                    self._allocate(seq_group)
                    seq_group.advance_prefill_range(max(seq_lens))
                    temp_num_batched_token +=  max(seq_lens) if seq_lens else 0
                    num_prompt += 1
                else:
                    assert False
                    seq_group.type = ""
                    self.priority_queues.push_front(seq_group)
                
        for seq_group in running:    
            if seq_group.type == "decode":
                if self.block_manager.can_append_slot(seq_group):
                    seq_group.lst_process_time = time.time()
                    running_with_prefill_start.append(seq_group)
                    self._append_slot(seq_group, blocks_to_copy)   
                    temp_num_batched_token += seq_group.num_seqs(status=SequenceStatus.RUNNING)
                else:
                    assert False
                    seq_group.type == ""
                    self._preempt(seq_group, blocks_to_swap_out, not_waiting=True)
                    self.priority_queues.push_front(seq_group)
                
        self.batch_queues = running_with_prefill_start

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=running_with_prefill_start,
            num_chunked_prefill_groups=0,
            num_prompt_groups=num_prompt,
            num_batched_tokens=temp_num_batched_token,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
            lora_enabled=self.lora_enabled,
            allow_both_swap=True
        )

        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()
        now = time.time()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
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
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        new_running = []
        for seq_group in self.real_running:
            if not seq_group.is_finished():
                 # put the request back to mlfq and try to demote it
                current_time = time.time()
                seq_group.process_time += (current_time - seq_group.lst_process_time)
                if seq_group.process_time > self.base_quantum * pow(
                    self.threshold, seq_group.get_priority()
                ):
                    seq_group.set_priority(seq_group.get_priority() + 1)
                    seq_group.process_time = 0
                self.priority_queues.push_front(seq_group) 
                new_running.append(seq_group)
                    
        self.real_running = new_running

        new_running = []
        for seq_group in self.running:
            if not seq_group.is_finished():
                new_running.append(seq_group)
        self.running = new_running

    def _allocate_and_set_running(self, seq_group: SequenceGroup,
                                  num_new_tokens: int) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
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
        not_waiting:bool = False,
    ) -> None:
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
