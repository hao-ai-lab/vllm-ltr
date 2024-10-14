##########################################
#Reproduce results in Table 3
##########################################

##########################################
#Learning to Rank
##########################################

#Lmsys/LTR/llama-70b Tau=0.62
python trainer.py --config configs/config_prefill_opt_350m.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

#ShareGPT/LTR/llama-70b Tau=0.55
python trainer.py --config configs/config_prefill_opt_350m.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

##########################################
#Classification
##########################################

#Lmsys/class bucket=100/llama-70b Tau=0.58
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket100-b32 --batch-size 32 --label-group-size 100 

#ShareGPT/class bucket=100/llama-70b Tau=0.49
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket100-b32 --batch-size 32 --label-group-size 100 


#Lmsys/class bucket=10/llama-70b Tau=0.57
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket10-b32 --batch-size 32 --label-group-size 10 

#ShareGPT/class bucket=10/llama-70b Tau=0.48
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket10-b32 --batch-size 32 --label-group-size 10 


#Lmsys/class bucket=1/llama-70b Tau=0.52
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket1-b32 --batch-size 32 --label-group-size 1 

#ShareGPT/class bucket=1/llama-70b Tau=0.28
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket1-b32 --batch-size 32 --label-group-size 1 


#Lmsys/class bucket=820/llama-70b Tau=0.21
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-lmsys-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 


#ShareGPT/class bucket=820/llama-70b Tau=0.18
python trainer.py --config configs/config_prefill_opt_350m_classify.txt --file jsonfiles/sharegpt-Meta-Llama-3-70B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-350m-llama3-70b-sharegpt-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 

##########################################
#Example for fine-tuning on 125M models
##########################################

#Lmsys/LTR/llama-8b Tau=0.64
python trainer.py --config configs/config_prefill_opt.txt --file jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-lmsys-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE

#ShareGPT/LTR/llama-8b Tau=0.52
python trainer.py --config configs/config_prefill_opt.txt --file jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32 --batch-size 32 --label-group-size 10 --loss listMLE


#Lmsys/class bucket=820/llama-70b acc: 0.97
python trainer.py --config configs/config_prefill_opt_classify.txt --file jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-lmsys-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 


#ShareGPT/class bucket=820/llama-70b acc: 0.92
python trainer.py --config configs/config_prefill_opt_classify.txt --file jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --job-dir MODEL --run-id opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32 --batch-size 32 --label-group-size 820 
