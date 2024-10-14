"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
import datasets
from tqdm import tqdm
import scipy
from fastchat.model import get_conversation_template
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
    shuffle_dataset: bool,
    start: int = 0,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    if dataset_path == "alpaca":
        dataset = datasets.load_dataset("tatsu-lab/alpaca",)['train']
        if shuffle_dataset:
            ds = ds.shuffle(seed=42)
        ds = dataset.select(range(start, start + int(num_requests * 1.2)))
        prompts = []
        for i, question in enumerate(ds):
            prompt = question['instruction'] + '\n' + question['input']
            messages = [
                {"role": "user", "content": prompt}
            ]
            conv = get_conversation_template(tokenizer.name_or_path)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
    elif dataset_path == "sharegpt":
        with open("ShareGPT_V3_unfiltered_cleaned_split.json") as f:
            dataset = json.load(f)
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        dataset = dataset[start:start + int(num_requests * 1.2)] 
        ds = dataset
        # Only keep the first two turns of each conversation.
        dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]
        prompts = []
        for prompt, _ in dataset:
            conv = get_conversation_template(tokenizer.name_or_path)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())
    elif dataset_path == "lmsys":
        dataset = datasets.load_dataset("lmsys/lmsys-chat-1m")['train']
        if shuffle_dataset:
            ds = ds.shuffle(seed=42)
        ds = dataset.select(range(start, start + int(num_requests * 1.2)))
        prompts = []
        for i, question in enumerate(ds):
            prompt = None
            for convsat in question['conversation']:
                if convsat['role'] == 'user':
                    prompt = convsat['content']
                    break
            if prompt is None:
                continue
            messages = [
                {"role": "user", "content": prompt}
            ]
            conv = get_conversation_template(tokenizer.name_or_path)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

    elif dataset_path == "openhermes":
        dataset = datasets.load_dataset("teknium/OpenHermes-2.5")['train']
        if shuffle_dataset:
            ds = ds.shuffle(seed=42)
        ds = dataset.select(range(start, start + int(num_requests * 1.2)))
        prompts = []
        for i, question in enumerate(ds):
            prompt = None
            for convsat in question['conversations']:
                if convsat['from'] == 'human':
                    prompt = convsat['value']
                    break
            if prompt is None:
                continue
            messages = [
                {"role": "user", "content": prompt}
            ]
            conv = get_conversation_template(tokenizer.name_or_path)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

    prompt_token_ids = tokenizer(prompts).input_ids
    tokenized_dataset = []
    for i in range(len(ds)):
        output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048000: #only filter too long prompt
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)

    return sampled_requests



def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    temperature: float,
    schedule_type: str,
    approx_portion: float,
    device: str,
    quantization_param_path: Optional[str],
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,

) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        schedule_type=schedule_type,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    if schedule_type == "fcfs-origin":
        pass
    elif schedule_type == "sjf":
        for rid, req in enumerate(requests):
            req = list(req)
            req[3] = req[2]
            requests[rid] = tuple(req)
    elif schedule_type == "approx-sjf":
        real = [req[2] for req in requests]
        ests = [req[2] for req in requests]
        rand_count = int(approx_portion * 1000)
        selected = random.sample(list(enumerate(ests)), rand_count)
        shuffled = [s[1] for s in selected]
        random.shuffle(shuffled)
        idx = 0
        for s in selected:
            ests[s[0]] = shuffled[idx] 
            idx += 1
        rela, p = scipy.stats.kendalltau(ests, real)
        print("kendall tau: ", rela, p)
        #shuffled = random.shuffle([s[1] for s in selected])
        for rid, req in enumerate(requests):
            req = list(req)
            req[3] = ests[rid]
            requests[rid] = tuple(req)
    elif schedule_type == "ljf":
        for rid, req in enumerate(requests):
            req = list(req)
            req[3] = - req[2]
            requests[rid] = tuple(req)
            
    # Add the requests to the engine.
    for prompt, _, output_len, est_tokens in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else temperature,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=False,
            max_tokens=output_len,
            est_tokens=est_tokens,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.perf_counter()
    # FIXME(woosuk): Do not use internal method.
    requests = llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    return end - start, requests


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve
    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len, args.shuffle_dataset, args.start)

    if args.backend == "vllm":
        elapsed_time, ret_requests = run_vllm(requests, args.model, args.tokenizer,
                                args.quantization, args.tensor_parallel_size,
                                args.seed, args.n, args.use_beam_search,
                                args.trust_remote_code, args.dtype,
                                args.max_model_len, args.enforce_eager,
                                args.kv_cache_dtype, args.temperature, args.schedule_type, args.approx_param, args.device, args.quantization_param_path, args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.gpu_memory_utilization,
            args.download_dir)

    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len, _ in requests)
    
    latencies = []
    for i in range(len(ret_requests)):
        latencies.append((ret_requests[i].latency, requests[i][2], requests[i][1]))

    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s, ")

    
    print("Sample Output", ret_requests[0])
    save_file_name = f"{args.dataset}-{args.model[args.model.rfind('/') + 1:]}-t{args.temperature}-s{args.seed}-l{args.output_len}-c{args.num_prompts if args.start == 0 else str(args.start) + ':' + str(args.start + args.num_prompts)}-r{args.shuffle_dataset}.jsonl"

    with open(save_file_name, "w") as outfile:
        for req in ret_requests:
            result_json = {"prompt":req.prompt, "generated":req.outputs[0].text}
            outfile.write(json.dumps(result_json) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.",
                       choices=["alpaca", "sharegpt", "openhermes", "lmsys"])
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=1024,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--schedule-type", type=str, default="fcfs",
                        choices=["fcfs","fcfs-origin", "sjf", "ljf", "approx-sjf", "approx-ljf"])
    parser.add_argument("--dir", type=str, default="THROUGHPUT")
    parser.add_argument("--approx-param", type=float, default=0)
    parser.add_argument("--shuffle-dataset", action="store_true")
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
