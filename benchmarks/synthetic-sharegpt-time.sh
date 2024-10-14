CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "PO-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 400 --dir SYNTHETIC --ignore-limit 


CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "opt-xxx-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 16 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json --dir SYNTHETIC --ignore-limit 


CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "fcfs-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 16 --dir SYNTHETIC --ignore-limit


CUDA_VISIBLE_DEVICES=0 python benchmark_throughput_original.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 10000 --schedule-type "tpt-class10-xxx-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 300 --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json --dir SYNTHETIC --ignore-limit 

