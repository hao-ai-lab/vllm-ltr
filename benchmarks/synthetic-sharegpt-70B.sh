python benchmark_throughput_original.py --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "PO-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 80 --dir SYNTHETIC --ignore-limit --tensor-parallel 8

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "opt-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 16 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "fcfs-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 16 --dir SYNTHETIC --ignore-limit --tensor-parallel 8 

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "tpt-class10-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 80 --prefill-predictor-model-config ./MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket820-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 

python benchmark_throughput_original.py --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "PO-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 80 --dir SYNTHETIC --ignore-limit --tensor-parallel 8 


python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "opt-xxx-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 16 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 


python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "fcfs-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 16 --dir SYNTHETIC --ignore-limit --tensor-parallel 8

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "tpt-class10-xxx-timelimit300"  --enable-chunked-prefill --enforce-eager --swap-space 80 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket820-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 




###########
#reproduce table 3
###########

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "tpt-class8192-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 80 --prefill-predictor-model-config ./MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket1-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 
python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "tpt-class82-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 80 --prefill-predictor-model-config ./MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket100-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 
python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "tpt-class820-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 80 --prefill-predictor-model-config ./MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket10-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 
python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 10000 --schedule-type "tpt-class164-xxx-synthetic((-1,-1,1000),)"  --enable-chunked-prefill --enforce-eager --swap-space 80 --prefill-predictor-model-config ./MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket50-b32/usage_config.json --dir SYNTHETIC --ignore-limit --tensor-parallel 8 
