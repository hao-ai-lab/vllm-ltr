"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple
import scipy
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from tqdm import tqdm
import scipy
import numpy as np
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS


def sample_requests(
    dataset_path: str,
    num_requests: int,
    ignore_limit: bool,
    output_len: int,
    output_len_def: dict,
    tokenizer: PreTrainedTokenizerBase,
    schedule_type: str,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    dataset = []
    with open(dataset_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)



    if "PO" in schedule_type:
        # Only keep the first two turns of each conversation.
        dataset = [(data["prompt"],
                    data["generated"], data["PO"]) for data in dataset]
        
        dataset = dataset[:int(num_requests*1.2)]
        assert len(dataset) >= num_requests
    
        # Tokenize the prompts and completions.
        prompts = [prompt for prompt, _, _ in dataset]
    else:
        # Only keep the first two turns of each conversation.
        dataset = [(data["prompt"],
                    data["generated"]) for data in dataset]
        
        dataset = dataset[:int(num_requests*1.2)]
        assert len(dataset) >= num_requests
    
        # Tokenize the prompts and completions.
        prompts = [prompt for prompt, _ in dataset]

    

    # Tokenize the prompts and completions.
    prompt_token_ids = tokenizer(prompts).input_ids
    if "FAKEPO" in schedule_type:
        completions = [completion for _, completion, _ in dataset]
        POs = [PO for _, _, PO in dataset]
    elif "PO" in schedule_type:
        completions = [completion for _, completion, _ in dataset]
        POs = [PO + 15 for _, _, PO in dataset]
    else:
        completions = [completion for _, completion in dataset]

    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    if output_len_def != -1:
        lens_box = []
        for k in output_len_def.keys():
            v = output_len_def[k]
            if v == -1:
                count = len(dataset) - len(lens_box)
            else:
                count = int(v * len(dataset))
            for vv in range(count):
                lens_box.append(k)
        random.shuffle(lens_box)

    for i in range(len(dataset)):
        if output_len_def != -1:
            use_output_len = lens_box[i]
            est_len = lens_box[i]
        elif "PO" in schedule_type:
            if output_len == -1:
                use_output_len = len(completion_token_ids[i])
            else:
                use_output_len = min(output_len, config.max_position_embeddings
                                     - len(prompt_token_ids[i]) - 1)
            est_len = POs[i]
        elif output_len == -1:
            use_output_len = len(completion_token_ids[i])
            est_len = len(completion_token_ids[i])
        else:
            use_output_len = output_len
            est_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], use_output_len,  est_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, use_output_len, prerun_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if not ignore_limit:
            if prompt_len < 4 or use_output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
                continue
            if prompt_len > 1024 or prompt_len + use_output_len > 20480:
            # Prune too long sequences.
                continue
        if "FAKEPO" not in schedule_type and "PO" in schedule_type:
            use_output_len += 15
        filtered_dataset.append((prompt, prompt_len, use_output_len, None, prerun_len))

    print("filter: ", len(filtered_dataset), num_requests, len(tokenized_dataset))
    assert len(filtered_dataset) >= num_requests
    # Sample the requests.
    #sampled_requests = random.sample(filtered_dataset, num_requests)
    return filtered_dataset[:num_requests]


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
    swap_space: int,
    gpu_memory_utilization: float,
    predictor_model_config: str,
    prefill_predictor_model_config: str,
    use_output_len: int,
    quantization_param_path: Optional[str],
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
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
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        schedule_type=schedule_type,
        swap_space=swap_space,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization_param_path=quantization_param_path,
        device=device,
        predictor_model_config=predictor_model_config,
        prefill_predictor_model_config=prefill_predictor_model_config,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    if schedule_type == "fcfs-origin":
        pass
    elif "finetuned-opt" in  schedule_type:
        from opt_predictor.modeling import CustomOPTModel
        pred_model = CustomOPTModel("facebook/opt-125m").half().to("cuda")
        pred_model.load_state_dict(torch.load('opt_predictor/model-1.pkl'))
        opt_tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m', trust_remote_code=True)
        scores = []
        real_length = []
        rid = 0
        for req in tqdm(requests):
            prompt = req[0]

            encoded_inputs = opt_tokenizer(prompt, max_length=1024, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
            #with torch.autocast(device_type="cuda"):
            outputs = pred_model(input_ids, attention_mask)

            predicted_scores = outputs.squeeze().tolist()            
            scores.append(-predicted_scores)
            real_length.append(req[4])


            req = list(req)
            req[3] = predicted_scores
            requests[rid] = tuple(req)
            rid += 1 

            #print("inp: ", output.hidden_states[-1].size(), inp_tokens, out)
        scores = np.array(scores)

        rela_s, p_s = scipy.stats.kendalltau(scores, real_length )
        print(f"real len {rela_s}, {p_s}")
        print(f"scores: {min(scores)}, {max(scores)}, {np.percentile(scores, 20):.2f} {np.percentile(scores, 50):.2f} {np.percentile(scores, 80):.2f} {np.percentile(scores, 90):.2f}")
    elif "PO" in schedule_type:
        for rid, req in enumerate(requests):
            req = list(req)
            req[3] = req[4]
            requests[rid] = tuple(req)
    elif schedule_type == "sjf" or schedule_type == "srtf" or schedule_type.startswith('predictor'):
        for rid, req in enumerate(requests):
            req = list(req)
            req[3] = req[2] if use_output_len == -1 else req[4]
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
    expected_generated_tokens = 0
    # Add the requests to the engine.
    for prompt, _, output_len, est_tokens, prerun_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=use_output_len == -1,
            max_tokens=output_len,
            est_tokens=est_tokens,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )
        expected_generated_tokens += output_len

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
    
        requests = sample_requests(dataset_path=args.dataset, num_requests=args.num_prompts, tokenizer=tokenizer,ignore_limit=args.ignore_limit,
                                   output_len=args.output_len,
                                   output_len_def=-1,
                                   schedule_type=args.schedule_type)

    if args.backend == "vllm":
        elapsed_time, ret_requests = run_vllm(requests, args.model, args.tokenizer,
                                args.quantization, args.tensor_parallel_size,
                                args.seed, args.n, args.use_beam_search,
                                args.trust_remote_code, args.dtype,
                                args.max_model_len, args.enforce_eager,
                                args.kv_cache_dtype, args.temperature, args.schedule_type, args.approx_param, args.device, args.swap_space, args.gpu_memory_utilization, predictor_model_config=args.predictor_model_config,prefill_predictor_model_config=args.prefill_predictor_model_config, use_output_len =args.output_len, quantization_param_path= args.quantization_param_path, enable_prefix_caching=args.enable_prefix_caching, enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_batched_tokens=args.max_num_batched_tokens, download_dir=args.download_dir)

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

    ototal_num_tokens = sum([len(req.outputs[0].token_ids) for req in ret_requests])
    #sum(output_len
    #                       for _, prompt_len, output_len, _, _ in requests)
    
    latencies = []
    lat = []
    nlatencies = []
    total_swap_out = 0
    total_swap_blocks = 0

    end_time = []
    real_length = []
    pred_scores = []
    aux_model_scores = []
    print("Example: ", ret_requests[0])
    real_ototal_num_tokens = 0
    for i in range(len(ret_requests)):
        #latency, output_len, prompt_len
        end_time.append(ret_requests[i].latency)
        real_length.append(len(ret_requests[i].outputs[0].token_ids)//32)
        pred_scores.append(ret_requests[i].outputs[0].pred_score)
        aux_model_scores.append(ret_requests[i].outputs[0].aux_model_score)
        latencies.append((ret_requests[i].latency,
                          len(ret_requests[i].outputs[0].token_ids),
                          requests[i][1], pred_scores[-1], ret_requests[i].outputs[0].aux_model_score, ret_requests[i].outputs[0].pred_score))
        if args.HOL:
            tmp = list(latencies[-1])
            tmp.append(ret_requests[i].HOL)
            latencies[-1] = tuple(tmp)
        nlatencies.append(ret_requests[i].latency / len(ret_requests[i].outputs[0].token_ids) )
        lat.append(ret_requests[i].latency)
        total_swap_out += ret_requests[i].outputs[0].running_info['swap_out']
        total_swap_blocks += ret_requests[i].outputs[0].running_info['swap_blocks']
        real_ototal_num_tokens += len(ret_requests[i].outputs[0].token_ids)
    if pred_scores[0] is not None:
        rela_s, p_s = scipy.stats.kendalltau(pred_scores, real_length )
        print(f"Pred Kendall Tau: {rela_s} {p_s}")
    if aux_model_scores[0] is not None:
        rela_s, p_s = scipy.stats.kendalltau(aux_model_scores, real_length )
        print(f"Aux Model Kendall Tau: {rela_s} {p_s}") 
    rela, p = scipy.stats.kendalltau(end_time, real_length )
    print(f"Finish Time Kendall Tau: {rela} {p}")
        #latencies = [(req.latency, ) for req in ret_requests]
    #outputtokens = []
    print(f"Throughput: {len(ret_requests) / elapsed_time:.2f} requests/s, "
          f"{real_ototal_num_tokens/elapsed_time:.2f} output tokens/s, ")
    print(f"Generated Tokens: {real_ototal_num_tokens}")
    print(f"Time: {elapsed_time:.2f} s, ")
    print(f"Total Swap: {total_swap_out}; Total Swap Blocks: {total_swap_blocks}")
    print(f"Norm Latency {len(nlatencies)}: P50 {np.percentile(nlatencies, 50):.2f} P80 {np.percentile(nlatencies, 80):.2f} P90 {np.percentile(nlatencies, 90):.2f} P95 {np.percentile(nlatencies, 95):.2f} P99 {np.percentile(nlatencies, 99):.2f} Mean {np.mean(nlatencies):.2f}")
    print(f"Latency {len(lat)}: P50 {np.percentile(lat, 50):.2f} P80 {np.percentile(lat, 80):.2f} P90 {np.percentile(lat, 90):.2f} P95 {np.percentile(lat, 95):.2f} P99 {np.percentile(lat, 99):.2f} Mean {np.mean(lat):.2f}")
    
    #torch.save( latencies, args.dir + "/" + f"latency-{args.schedule_type}-{args.approx_param}.pt")
    torch.save(latencies, args.dir + "/" + f"latency-{args.schedule_type}-{args.model[args.model.rfind('/')+1:]}-p{args.approx_param}-r{len(nlatencies)}-o{args.output_len}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=-1,
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
    parser.add_argument("--ignore-limit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--swap-space", type=int, default=30)
    parser.add_argument("--schedule-type", type=str, default="fcfs-origin" ) #, choices=["fcfs", "sjf", "ljf", "approx-sjf", "approx-ljf", "mlfq", "mlfq-async"])
    parser.add_argument("--predictor-model-config", type=str, default="")
    parser.add_argument("--prefill-predictor-model-config", type=str, default="")
    parser.add_argument("--dir", type=str, default="THROUGHPUT")
    parser.add_argument("--approx-param", type=float, default=0)
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
    parser.add_argument("--HOL",
                        action="store_true",
                        help="enforce HOL PLOT")
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
