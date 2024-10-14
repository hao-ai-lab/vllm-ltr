# Reproducing Predictor Training

### Step 1: Download the Dataset

To download the required dataset, use the following command:

```bash
huggingface-cli download LLM-ltr/Llama3-Trace --local-dir jsonfiles --repo-type dataset
```

### Step 2: Train the Models

To initiate model training, simply run:

```bash
bash train.sh
```

The trained models will be stored in the `MODEL/results` directory.

### Hardware Requirements

- The **350M predictor** requires **80GB of GPU memory**.
- The **125M predictor** requires **40GB of GPU memory**.
