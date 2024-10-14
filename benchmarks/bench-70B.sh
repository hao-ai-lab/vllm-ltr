#fcfs
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --swap-space 16 --disable-log-requests --schedule-type fcfs --enable-chunked-prefill --enforce-eager --tensor-parallel 8 &
sleep 120

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 2 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 4 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 8 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 16 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 32 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 64 --result-dir RESULTS 

kill $!
pkill python
sleep 60


#PO-X
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --swap-space 40 --disable-log-requests --schedule-type PO --enable-chunked-prefill --enforce-eager  --tensor-parallel 8 &
sleep 200

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 2 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 4 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 8 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 16 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 32 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset PO-gen-sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 64 --result-dir RESULTS 

kill $!
pkill python
sleep 60


#opt-ours
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --swap-space 16 --disable-log-requests --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32/usage_config.json --tensor-parallel 8 &
sleep 120

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 2 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 4 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 8 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 16 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 32 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 64 --result-dir RESULTS 

kill $!
pkill python
sleep 60

#mlfq
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct  --disable-log-requests --schedule-type mlfq-base0.03-thres10 --enable-chunked-prefill --enforce-eager --swap-space 60 --tensor-parallel 8 &
sleep 180

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 2 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 4 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 8 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 16 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 32 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 64 --result-dir RESULTS 

kill $!
pkill python
sleep 60

#opt-class10
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --disable-log-requests --schedule-type tpt-class10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config ./MODEL/results/opt-350m-llama3-70b-sharegpt-class-trainbucket820-b32/usage_config.json --swap-space 40 --tensor-parallel 8 &
sleep 120

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 2 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 4 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 8 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 16 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 32 --result-dir RESULTS 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 64 --result-dir RESULTS 

kill $!
pkill python
sleep 60

