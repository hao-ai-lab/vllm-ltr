###########
#reproduce table 1
###########

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type mlfq-quant0.03-thres10 --enable-chunked-prefill --enforce-eager --swap-space 60 --dir BURST --tensor-parallel 8 

python benchmark_throughput_original.py --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type PO  --enable-chunked-prefill --enforce-eager --swap-space 60 --dir BURST --tensor-parallel 8 

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type fcfs  --enable-chunked-prefill --enforce-eager --swap-space 16 --dir BURST --tensor-parallel 8 

#ours
python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type opt-xxx  --enable-chunked-prefill --enforce-eager --swap-space 16 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32/usage_config.json --dir BURST --tensor-parallel 8

#class10
python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type tpt-class10-xxx  --enable-chunked-prefill --enforce-eager --swap-space 60 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket820-b32/usage_config.json --dir BURST --tensor-parallel 8 


###########
#reproduce table 3
###########
python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type tpt-class820-xxx  --enable-chunked-prefill --enforce-eager --swap-space 60 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket10-b32/usage_config.json --dir BURST --tensor-parallel 8 

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type tpt-class82-xxx  --enable-chunked-prefill --enforce-eager --swap-space 60 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket100-b32/usage_config.json --dir BURST --tensor-parallel 8 

python benchmark_throughput_original.py --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --num-prompts 2000 --schedule-type tpt-class8192-xxx  --enable-chunked-prefill --enforce-eager --swap-space 60 --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket1-b32/usage_config.json --dir BURST --tensor-parallel 8 
