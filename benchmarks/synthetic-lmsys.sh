CUDA_VISIBLE_DEVICES=1 python benchmark_throughput_original.py --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "PO-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 200 --dir SYNTHETIC --ignore-limit


CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "opt-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 16 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json --dir SYNTHETIC --ignore-limit 

CUDA_VISIBLE_DEVICES=1 python benchmark_throughput_original.py --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "fcfs-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 16 --dir SYNTHETIC --ignore-limit

CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "tpt-class10-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 100 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json --dir SYNTHETIC --ignore-limit 

