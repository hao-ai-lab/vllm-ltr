
CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type PO  --enable-chunked-prefill --enforce-eager --swap-space 100 --dir BURST 

CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type mlfq-quant0.03-thres2 --enable-chunked-prefill --enforce-eager --swap-space 200 --dir BURST 


CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type opt-xxx  --enable-chunked-prefill --enforce-eager --swap-space 16 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json --dir BURST 

CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type fcfs  --enable-chunked-prefill --enforce-eager --swap-space 16 --dir BURST 

CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 2000 --schedule-type tpt-class10-xxx  --enable-chunked-prefill --enforce-eager --swap-space 100 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json --dir BURST 
