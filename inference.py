import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a fine-tuned LLM")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of the fine-tuned model')
    parser.add_argument('--input_text', type=str, required=True, help='Input text for inference')
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    inputs = tokenizer(args.input_text, return_tensors='pt')
    outputs = model.generate(**inputs)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")

if __name__ == "__main__":
    main()
