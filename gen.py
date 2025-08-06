import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from safetensors.torch import load_file
from datasets import load_dataset
from tqdm import tqdm

# Configuration
MODEL_PATH = "/shared_workspace/alex/sft-test/LLaMA-Factory/saves/qwen2_5coder-7b/full/sft"  # Path to directory containing safetensors and config
DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
MAX_INPUT_LENGTH = 20000
MAX_TARGET_LENGTH = 10000
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load model from safetensors
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="test")
    
    # Generate predictions
    results = []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i:i+BATCH_SIZE]
        # Prepare inputs
        inputs = [
            f"Fix issue in {repo} at {base_commit}: {problem_statement}\nContext: {hint_text}"
            for repo, base_commit, problem_statement, hint_text in zip(
                batch["repo"],
                batch["base_commit"],
                batch["problem_statement"],
                batch["hint_text"]
            )
        ]
        
        # Tokenize
        tokenized = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                max_length=MAX_TARGET_LENGTH,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode predictions
        predictions = tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        # Collect results
        for j, instance_id in enumerate(batch["instance_id"]):
            results.append({
                "instance_id": instance_id,
                "model_patch": predictions[j]
            })

    # Save results
    with open("predictions.jsonl", "w") as f:
        for res in results:
            f.write(f"{json.dumps(res)}\n")

if __name__ == "__main__":
    import json
    main()