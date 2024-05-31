# Fine-Tune LLM Framework

This repository provides a framework for fine-tuning large language models (LLMs) such as GPT-2, LLAMA, and Mistral. The framework is designed to be flexible and easy to use, allowing users to specify the pretrained model, dataset, and optimization options via command-line arguments.

## Features
- Support for popular LLMs: GPT-2, LLAMA, Mistral
- Command-line interface for easy configuration
- Support for optimization methods: LoRA, QLORA

## Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/finetune-llm-framework.git
cd finetune-llm-framework
pip install -r requirements.txt
```

## Usage
Fine-tune a pretrained model with a specified dataset:

```bash
python src/main.py --model_name gpt2 --dataset_name wikitext --output_dir ./results --epochs 3 --batch_size 8 --learning_rate 5e-5 --optimization lora
```

Run inference on the fine-tuned model:

```bash
python src/inference.py --model_dir ./results --input_text "Once upon a time"
```

## Arguments for Training
--model_name: Pretrained model name (e.g., gpt2, llama, mistral)
--dataset_name: Dataset name (e.g., wikitext, glue)
--output_dir: Directory to save the results
--epochs: Number of training epochs
--batch_size: Batch size for training
--learning_rate: Learning rate
--optimization: Optimization method (lora, qlora, none)

## Arguments for Inference
--model_dir: Directory of the fine-tuned model
--input_text: Input text for inference

## Example

To fine-tune the GPT-2 model on the Wikitext dataset using LoRA optimization:
```bash
python src/main.py --model_name gpt2 --dataset_name wikitext --output_dir ./results --epochs 3 --batch_size 8 --learning_rate 5e-5 --optimization lora
```

To run inference on the fine-tuned model:
```bash
python src/inference.py --model_dir ./results --input_text "Once upon a time"
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
