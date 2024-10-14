"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
"""
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple
import scipy
import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase, AutoConfig

from vllm.transformers_utils.tokenizer import get_tokenizer

import torch

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float


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
        
    prompt_token_ids = tokenizer(prompts).input_ids
    if "PO" in schedule_type:
        completions = [completion for _, completion, _ in dataset]
        POs = [PO for _, _, PO in dataset]
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

    config = AutoConfig.from_pretrained(tokenizer.name_or_path)

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
            use_output_len = min(output_len, config.max_position_embeddings - len(prompt_token_ids[i]) - 1)
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
        if "PO" in schedule_type:
            use_output_len += 15
        filtered_dataset.append((prompt, prompt_len, use_output_len, None, prerun_len))
    
    assert len(filtered_dataset) >= num_requests
    # Sample the requests.
    #sampled_requests = random.sample(filtered_dataset, num_requests)
    return filtered_dataset[:num_requests]


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    cv: float
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        shape = 1 / (cv * cv)
        scale = cv * cv / request_rate
        interval = np.random.gamma(shape, scale)

        # Sample the request interval from the exponential distribution.
        #interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    total_input = 0
    completed = 0
    tpots = []
    ttfts = []
    latencies = []
    nlatencies = []
    input_lens = []
    est_lens = []
    pred_scores = []
    aux_model_scores = []
    texts = []
    
    for req in input_requests:
        est_lens.append(req[3])
    
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer(outputs[i].generated_text).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            ttfts.append(outputs[i].ttft)
            completed += 1
            latencies.append(outputs[i].latency)
            nlatencies.append(outputs[i].latency / output_len)
            input_lens.append(input_requests[i][1])
            pred_scores.append(outputs[i].pred_score)
            aux_model_scores.append(outputs[i].aux_model_score)

            texts.append((input_requests[i][0], outputs[i].generated_text))
        else:
            actual_output_lens.append(0)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots) * 1000,
        median_tpot_ms=np.median(tpots) * 1000,
        p99_tpot_ms=np.percentile(tpots, 99) * 1000,
    )

    return metrics, (ttfts, tpots, latencies, nlatencies, actual_output_lens, input_lens, est_lens, pred_scores, aux_model_scores, texts)


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    save_dir: str,
    schedule_type: str,
    approx_type: str,
    approx_param: float,
    rate: float,
    cv: float,
    output_len: int,
    output_len_def: dict,
    eval_max_tpot: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if schedule_type == "fcfs" or schedule_type == "fcfs-origin" or schedule_type.startswith("sjf-predictor-") or schedule_type.startswith("mlfq") or schedule_type.startswith("predictor-") or schedule_type.startswith("opt") or schedule_type.startswith("xpt") or schedule_type.startswith("tpt"):

        pass
    elif schedule_type == "sjf" or schedule_type.startswith("srtf") or schedule_type.startswith("sjf-preempt-") or schedule_type.startswith("sjf-ranking-") or schedule_type.startswith("sjf-file-") : 
        #only work when output_len==-1
        if schedule_type.startswith("sjf-file-"):
            ee = torch.load(schedule_type[len("sjf-file-"):])
            for rid, req in enumerate(input_requests):
                req = list(req)
                req[3] = -ee[rid]
                input_requests[rid] = tuple(req)
        else:
            for rid, req in enumerate(input_requests):
                req = list(req)
                req[3] = req[4]
                input_requests[rid] = tuple(req)
    elif schedule_type.startswith("approx-sjf") or schedule_type.startswith("approx-sjf-preempt-") or schedule_type.startswith("approx-sjf-ranking-"):
        #only work when output_len==-1
        assert output_len == -1
        real = [req[4] for req in input_requests]
        ests = [req[4] for req in input_requests]
        rand_count = int(approx_param * len(input_requests))
        
        sest = list(enumerate(ests))
        sorted_ests = sorted(sest, key=lambda x:x[1])
        
        if approx_type == "full":        
            selected = random.sample(list(enumerate(ests)), rand_count)
        elif approx_type == "head":
            selected = sorted_ests[:rand_count]  #random.sample(list(enumerate(ests) ), rand_count)
        elif approx_type == "tail":
            selected = sorted_ests[-rand_count:]  
        elif approx_type == "middle":
            offset = (len(ests) - rand_count) // 2
            selected = sorted_ests[offset:offset + rand_count]  
        elif approx_type == "hat":
            selected = sorted_ests[:rand_count // 2] + sorted_ests[-rand_count // 2:]
            
            
        shuffled = [s[1] for s in selected]
        random.shuffle(shuffled)
        idx = 0
        for s in selected:
            ests[s[0]] = shuffled[idx] 
            idx += 1
        rela, p = scipy.stats.kendalltau(ests, real)
        print("kendall tau: ", rela, p)
        #shuffled = random.shuffle([s[1] for s in selected])
        for rid, req in enumerate(input_requests):
            req = list(req)
            req[3] = ests[rid]
            input_requests[rid] = tuple(req)
    elif schedule_type == "ljf":
        #only work when output_len==-1
        for rid, req in enumerate(input_requests):
            req = list(req)
            req[3] = - req[4]
            input_requests[rid] = tuple(req)
    else:
        assert False, f"Schedule Type {schedule_type} Not Supported"


    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    print(f"Traffic request rate: {request_rate}; Coefficient Variation: {cv}; Output len: {output_len}")

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate, cv):
        prompt, prompt_len, use_output_len, est_tokens, prerun_len = request
        #print("igNORE: ", output_len == -1, eval(output_len_def) != -1, output_len == -1 or eval(output_len_def) != -1, output_len_def)
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=use_output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
            est_tokens=est_tokens,
            ignore_eos=output_len == -1 or eval(output_len_def) != -1,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, (ttfts, real_tpots, latencies, nlatencies, actual_output_lens, input_lens, est_lens, pred_scores, aux_model_scores, texts) = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s='Time Normalized Latency',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean Nlatency (ms):", np.mean(nlatencies) * 1000))
    print("{:<40} {:<10.2f}".format("Median Nlatency (ms):",
                                    np.median(nlatencies) * 1000))
    print("{:<40} {:<10.2f}".format("P80 Nlatency (ms):", np.percentile(nlatencies, 80) * 1000))
    print("{:<40} {:<10.2f}".format("P90 Nlatency (ms):", np.percentile(nlatencies, 90) * 1000))
    print("{:<40} {:<10.2f}".format("P95 Nlatency (ms):", np.percentile(nlatencies, 95) * 1000))
    print("{:<40} {:<10.2f}".format("P99 Nlatency (ms):", np.percentile(nlatencies, 99) * 1000))
    print("=" * 50)

    if est_lens[0] is not None:
        tau, p = scipy.stats.kendalltau(est_lens, actual_output_lens)
        print(f"Est Kendall's Tau: {tau} {p}")
    if pred_scores[0] is not None:
        tau, p = scipy.stats.kendalltau(pred_scores, actual_output_lens)
        print(f"Pred Kendall's Tau: {tau} {p}")
    if aux_model_scores[0] is not None:
        tau, p = scipy.stats.kendalltau(aux_model_scores, actual_output_lens)
        print(f"Aux Model Kendall's Tau: {tau} {p}")



    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    if eval_max_tpot:
        max_tpots = []
        for i in range(len(outputs)):
            max_tpot = max(outputs[i].ttft, max(outputs[i].itl))
            max_tpots.append(max_tpot)
        print("{s:{c}^{n}}".format(s='MAX-TPOT',
                                   n=50,
                                   c='-'))
        print("{:<40} {:<10.2f}".format("Mean MAX-TPOT (ms):", np.mean(max_tpots) * 1000))
        print("{:<40} {:<10.2f}".format("Median MAX-TPOT (ms):",
                                        np.median(max_tpots) * 1000))
        print("{:<40} {:<10.2f}".format("P80 MAX-TPOT (ms):", np.percentile(max_tpots, 80) * 1000))
        print("{:<40} {:<10.2f}".format("P90 MAX-TPOT (ms):", np.percentile(max_tpots, 90) * 1000))
        print("{:<40} {:<10.2f}".format("P95 MAX-TPOT (ms):", np.percentile(max_tpots, 95) * 1000))
        print("{:<40} {:<10.2f}".format("P99 MAX-TPOT (ms):", np.percentile(max_tpots, 99) * 1000))
        print("=" * 50)  
        torch.save([ttfts, real_tpots, latencies, nlatencies, actual_output_lens, input_lens, est_lens, texts, aux_model_scores, pred_scores, [output.itl for output in outputs]], save_dir + "/" + f"latency-{schedule_type}-{model_id[model_id.rfind('/')+1:]}-p{approx_param}-r{rate}-c{cv}-t{len(input_requests) / request_rate}-o{output_len if eval(output_len_def) == -1 else output_len_def}{'' if approx_type == 'full' else '-' + approx_type}.pt")
    else:
        torch.save([ttfts, real_tpots, latencies, nlatencies, actual_output_lens, input_lens, est_lens, texts, aux_model_scores, pred_scores], save_dir + "/" + f"latency-{schedule_type}-{model_id[model_id.rfind('/')+1:]}-p{approx_param}-r{rate}-c{cv}-t{len(input_requests) / request_rate}-o{output_len if eval(output_len_def) == -1 else output_len_def}{'' if approx_type == 'full' else '-' + approx_type}.pt")
    print("save to: ", save_dir + "/" + f"latency-{schedule_type}-{model_id[model_id.rfind('/')+1:]}-p{approx_param}-r{rate}-c{cv}-t{len(input_requests) / request_rate}-o{output_len if eval(output_len_def) == -1 else output_len_def}{'' if approx_type == 'full' else '-' + approx_type}.pt")
    
    print()
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)
    if args.num_prompts == -1:
        assert args.request_time != -1
        args.num_prompts = args.request_time * args.request_rate
    args.num_prompts = int(args.num_prompts)
    input_requests = sample_requests(args.dataset, args.num_prompts, args.ignore_limit, args.output_len, eval(args.output_len_def), tokenizer, args.schedule_type)

    
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            save_dir=args.result_dir,
            schedule_type=args.schedule_type,
            approx_type=args.approx_type,
            approx_param=args.approx_param,
            rate=args.request_rate,
            cv=args.cv,
            output_len=args.output_len,
            output_len_def=args.output_len_def,
            eval_max_tpot=args.eval_max_tpot,
        ))

    # Save config and results to json
    if True:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["schedule_type"] = args.schedule_type
        result_json["approx_type"] = args.approx_type
        result_json["approx_param"] = args.approx_param
        result_json["output_len"] = args.output_len
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-cv{args.cv}-{base_model_id}-{args.schedule_type}{'-p'+ str(args.approx_param) if args.approx_param else ''}-{current_dt}.json"
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--eval-max-tpot", action="store_true")
    parser.add_argument("--output-len",
                        type=int,
                        default=-1,
                        help="If output-len is -1, we will use genlen in dataset, or we will use outputlen as max-output-len")
    parser.add_argument("--output-len-def",
                        type=str, #{1:0.5, 23:0.5, 21:-1} 50%1, 50% 23 , others 21
                        default="-1",
                        help="If output-len-def is -1, we will ignore or we will use it")
    parser.add_argument(
        "--request-time",
        type=float,
        default=-1.0,
        help=""
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--ignore-limit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument("--schedule-type", type=str, default="fcfs-origin") #, choices=["fcfs", "sjf", "ljf", "approx-sjf", "approx-ljf", "sjf-preempt", "approx-sjf-preempt"])
    parser.add_argument("--approx-type", type=str, default="full", choices=['full', 'middle', 'head', 'tail', 'hat']) 
    parser.add_argument("--approx-param", type=float, default=0)
    parser.add_argument("--cv", type=float, default=1.0)
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="SERVE",
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )

    args = parser.parse_args()
    main(args)
