#set +e
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --swap-space 16 --disable-log-requests --schedule-type fcfs --enable-chunked-prefill --enforce-eager --tensor-parallel 8 &
sleep 120

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 2 --eval-max-tpot --result-dir FAIR 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 4 --eval-max-tpot --result-dir FAIR 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 8 --eval-max-tpot --result-dir FAIR 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 16 --eval-max-tpot --result-dir FAIR 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 32 --eval-max-tpot --result-dir FAIR

kill $!
pkill python
sleep 60



python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --swap-space 40 --disable-log-requests --schedule-type opt-xxx-starv200-period10 --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32/usage_config.json --tensor-parallel 8 &
sleep 120

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx-starv200-period10 --output-len -1 --request-rate 2 --result-dir FAIR  --eval-max-tpot 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx-starv200-period10 --output-len -1 --request-rate 4 --result-dir FAIR  --eval-max-tpot 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx-starv200-period10 --output-len -1 --request-rate 8 --result-dir FAIR  --eval-max-tpot 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx-starv200-period10 --output-len -1 --request-rate 16 --result-dir FAIR  --eval-max-tpot 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx-starv200-period10 --output-len -1 --request-rate 32 --result-dir FAIR  --eval-max-tpot

kill $!
pkill python
sleep 60
#
#kill $!
#pkill python
#sleep 60
#opt-ours
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --swap-space 16 --disable-log-requests --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32/usage_config.json --tensor-parallel 8 &
sleep 120

python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 2 --result-dir FAIR  --eval-max-tpot 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 4 --result-dir FAIR  --eval-max-tpot
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 8 --result-dir FAIR  --eval-max-tpot
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 16 --result-dir FAIR  --eval-max-tpot 
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-70B-Instruct  --tokenizer meta-llama/Meta-Llama-3-70B-Instruct --dataset sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 32 --result-dir FAIR  --eval-max-tpot 

kill $!
pkill python
sleep 60



