
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type fcfs --enable-chunked-prefill --enforce-eager &
sleep 60
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 2
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 4 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 8 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 16
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 32
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 64 

kill $!
sleep 60



#PO new
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 200 --disable-log-requests --schedule-type PO --enable-chunked-prefill --enforce-eager  &
sleep 120 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 2 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 4 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 8 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 16 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 32 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 64

kill $!
sleep 60



#ours
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json &
sleep 60
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 2 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 4 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 8 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 16 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 32 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 64 

kill $!
sleep 60



#mlfq
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type mlfq-base0.03-thres10 --enable-chunked-prefill --enforce-eager --swap-space 200 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 2 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 4 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 8 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 16
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 32 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 64 

kill $!
sleep 60


#opt-class10
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type tpt-class10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json --swap-space 100 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 2 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 4 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 8 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 16 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 32 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 64

kill $!
sleep 60


