# Reproducing Results

> **Note:** Serving the 70B model requires 8 A100 40G GPUs, while the 8B model requires 1 A100 40G GPU.

## Generate Serving Data

To download pre-generated data, run:

```bash
huggingface-cli download LLM-ltr/Llama3-Trace --local-dir ./Llama3-Trace --repo-type dataset
mv Llama3-Trace/*.jsonl .
```

To download ShareGPT data, run:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Alternatively, generate the data from scratch using:

```bash
# Train and Eval data for lmsys dataset
python benchmark_generate_dataset.py --dataset lmsys --model meta-llama/Meta-Llama-3-8B-Instruct --schedule-type fcfs-origin --temperature 1.0 --num-prompts 10000 --seed 0 --output-len 8192 --start 20000 # Train
python benchmark_generate_dataset.py --dataset lmsys --model meta-llama/Meta-Llama-3-8B-Instruct --schedule-type fcfs-origin --temperature 1.0 --num-prompts 10000 --seed 0 --output-len 8192 # Eval

# Train and Eval data for sharegpt dataset
python benchmark_generate_dataset.py --dataset sharegpt --model meta-llama/Meta-Llama-3-8B-Instruct --schedule-type fcfs-origin --temperature 1.0 --num-prompts 10000 --seed 0 --output-len 8192 --start 20000 # Train
python benchmark_generate_dataset.py --dataset sharegpt --model meta-llama/Meta-Llama-3-8B-Instruct --schedule-type fcfs-origin --temperature 1.0 --num-prompts 10000 --seed 0 --output-len 8192 # Eval

# Generate data for 70B model
python benchmark_generate_dataset.py --dataset sharegpt --model meta-llama/Meta-Llama-3-70B-Instruct --schedule-type fcfs-origin --temperature 1.0 --num-prompts 10000 --enforce-eager --swap-space 16 --tensor-parallel 8 --seed 0 --output-len 8192
```

To generate the PO dataset (with LM's prediction length label), use:

```bash
python benchmark_append_dataset_PO.py --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --seed 0 --schedule-type fcfs
```

## Download Pre-trained Models

For model downloads, you can simply use:

```bash
mkdir -p MODEL/results
huggingface-cli download LLM-ltr/OPT-Predictors --local-dir MODEL/results
```

## Reproducing Results

### Reproduce Table 1

```bash
mkdir BURST
bash burst-lmsys.sh        # lmsys/8B
bash burst-sharegpt.sh     # sharegpt/8B
bash burst-lmsys-70B.sh    # lmsys/70B
bash burst-sharegpt-70B.sh # sharegpt/70B
```

### Reproduce Figure 3

```bash
mkdir RESULTS 
bash bench.sh         # sharegpt/8B
bash bench-lmsys.sh   # lmsys/8B
bash bench-70B.sh     # sharegpt/70B
bash bench-lmsys-70B.sh  # lmsys/70B
```

### Reproduce Figures 4 & 5

```bash
mkdir FAIR
bash fair-lmsys-70B.sh    # lmsys/70B
bash fair-sharegpt-70B.sh # sharegpt/70B
```

### Reproduce Table 2

```bash
mkdir SYNTHETIC
bash synthetic-lmsys-time.sh
bash synthetic-sharegpt-time.sh
bash synthetic-lmsys.sh
bash synthetic-sharegpt.sh
bash synthetic-sharegpt-70B.sh
bash synthetic-lmsys-70B.sh
```

### Reproduce Table 3

Check these scripts to find the relevant parts for Table 3:

```bash
bash burst-lmsys-70B.sh        # lmsys/70B
bash burst-sharegpt-70B.sh     # sharegpt/70B
bash synthetic-sharegpt-70B.sh
bash synthetic-lmsys-70B.sh
```
