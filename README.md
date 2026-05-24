# VLM Action Captioning

Fine-tuned Qwen2-VL-2B for video action understanding and captioning, 
with full MLOps pipeline.

## Architecture

[diagram placeholder - add after Week 2]

## Results

| Model | CIDEr | METEOR | Latency (p50) |
|---|---|---|---|
| Qwen2-VL-2B base | - | - | - |
| Qwen2-VL-2B QLoRA | - | - | - |

## Stack

Training: PyTorch, PEFT/QLoRA, Weights and Biases, Kaggle T4  
Serving: vLLM, FastAPI, Modal.com serverless GPU  
CI/CD: GitHub Actions, Docker, ECR  
Demo: Hugging Face Spaces, Gradio  
Registry: AWS SageMaker Model Registry  

## Quickstart

```bash
pip install -e ".[train,serve,dev]"
make train-local   # smoke test on 50 samples
make serve-local   # run inference API locally
```

## Reproducing training

[fill after Week 2]

## Deployment

[fill after Week 3]
