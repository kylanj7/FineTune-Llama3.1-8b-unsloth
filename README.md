# LoRA Fine-Tuning Pipeline

A streamlined pipeline for fine-tuning large language models using Low-Rank Adaptation (LoRA) with Unsloth optimization.

## Overview

This pipeline leverages [Unsloth](https://github.com/unslothai/unsloth) to achieve up to 2x faster fine-tuning speeds with 80% less VRAM usage compared to standard fine-tuning approaches. The implementation supports QLoRA (4-bit quantization) for efficient training on consumer hardware.

## Features

- **Memory Efficient**: 4-bit quantization with QLoRA reduces VRAM requirements significantly
- **Fast Training**: Unsloth's optimized kernels provide 2-10x speedup over standard implementations
- **Flexible Configuration**: Easily adjustable hyperparameters for different models and tasks
- **HuggingFace Compatible**: Fully compatible with HuggingFace transformers and TRL ecosystem
- **Multiple Export Formats**: Save as LoRA adapters, merged models, or GGUF format

## Requirements

### Hardware
- NVIDIA GPU (tested on RTX 3090)
- CUDA-compatible drivers

### Software
```bash
# Core dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

## Quick Start

### 1. Load Model with Unsloth

```python
from unsloth import FastLanguageModel

max_seq_length = 2048  # Supports automatic RoPE Scaling
dtype = None  # Auto-detect: Float16 for T4/V100, BFloat16 for Ampere+

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,  # Enable 4-bit quantization
)
```

### 2. Configure LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### 3. Train

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer.train()
```

### 4. Save and Export

```python
# Save LoRA adapter
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Or save merged 16-bit model
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Export to GGUF for llama.cpp
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
```

## Configuration

### Key Hyperparameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `r` | LoRA rank | 8, 16, 32, 64 |
| `lora_alpha` | LoRA scaling factor | Same as rank |
| `max_seq_length` | Maximum sequence length | 512, 1024, 2048 |
| `learning_rate` | Learning rate | 1e-4 to 5e-4 |
| `batch_size` | Training batch size | 2-8 (depends on VRAM) |

### Memory Optimization Tips

1. **Reduce sequence length**: Lower `max_seq_length` for VRAM constraints
2. **Adjust batch size**: Use gradient accumulation for effective larger batches
3. **Enable gradient checkpointing**: Trades compute for memory
4. **Use 4-bit quantization**: `load_in_4bit=True` reduces memory by ~75%

## Supported Models

Unsloth provides optimized 4-bit and GGUF versions of popular models:
- Llama 3.x (all sizes)
- Mistral 7B
- Qwen 2.5
- Gemma 2
- Phi-3.5
- DeepSeek Coder
- And many more on [HuggingFace](https://huggingface.co/unsloth)

## Inference

```python
# Fast inference with Unsloth
FastLanguageModel.for_inference(model)

inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## References

- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **Unsloth HuggingFace**: https://huggingface.co/unsloth
- **Unsloth Documentation**: https://docs.unsloth.ai/
- **TRL Integration Guide**: https://huggingface.co/docs/trl/en/unsloth_integration
- **Original Blog Post**: https://huggingface.co/blog/unsloth-trl

## Citation

```bibtex
@software{unsloth2024,
  author = {Daniel Han and Michael Han},
  title = {Unsloth: Fast LLM Fine-tuning},
  year = {2024},
  url = {https://github.com/unslothai/unsloth}
}
```

## License

This pipeline follows the licensing of the underlying models and Unsloth library. Please refer to individual model licenses on HuggingFace.

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_seq_length`
- Lower `per_device_train_batch_size`
- Enable gradient checkpointing
- Use smaller LoRA rank

### Slow Training
- Ensure `load_in_4bit=True` is set
- Use Unsloth-optimized models (ending in `-bnb-4bit`)
- Check CUDA drivers are up to date

### Installation Issues
- Create isolated conda/venv environment
- Verify PyTorch CUDA compatibility
- Check Unsloth's supported PyTorch versions

## Acknowledgments

Built with [Unsloth](https://huggingface.co/unsloth) by Daniel Han and Michael Han - making LLM fine-tuning accessible to everyone.
