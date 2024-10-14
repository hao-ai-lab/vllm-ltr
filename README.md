# Efficient LLM Scheduling by Learning to Rank [[paper](https://arxiv.org/abs/2408.15792)]

vllm-ltr is an efficient serving system that approximates Shortest Job First (SJF) scheduling using learning to rank.

## Motivation
Most Large Language Model (LLM) serving systems use a First-Come-First-Serve (FCFS) strategy due to the unpredictable output length of requests, which leads to Head-Of-Line (HOL) blocking and reduced performance. While predicting exact output lengths is challenging, we show that it’s possible to rank requests based on their relative output lengths using learning to rank. This ranking enables more efficient scheduling. Our novel scheduler improves upon traditional methods by better approximating SJF, leading to substantial performance gains, such as a 2.8x reduction in latency for chatbot services and a 6.5x increase in throughput for synthetic data generation.

## Installation

vllm-ltr is built on [vLLM](https://github.com/vllm-project/vllm). To install our modified version, follow these steps:

```
conda create -n vllm-ltr python=3.10
conda activate vllm-ltr
git clone https://github.com/hao-ai-lab/vllm-ltr.git
cd vllm-ltr
pip install -e .  
```

## Reproduce Results

For predictor training, refer to the `./train` directory, and for end-to-end evaluation, check the `./benchmark` directory.

Fine-tuned predictors can be found on [huggingface](https://huggingface.co/LLM-ltr/OPT-Predictors).

## Citation
```
@article{fu2024efficient,
  title={Efficient LLM Scheduling by Learning to Rank},
  author={Fu, Yichao and Zhu, Siqi and Su, Runlong and Qiao, Aurick and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2408.15792},
  year={2024}
}
```