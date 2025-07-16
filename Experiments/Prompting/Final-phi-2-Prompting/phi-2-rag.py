import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import re
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Specify GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load the F* dataset
print("Loading F* dataset...")
dataset = load_dataset("microsoft/FStarDataSet")
train_data = dataset["train"]  # Limit to 2000 samples
eval_data = dataset["validation"]  # Limit to 100 samples
test_data = dataset["test"] # Limit to 10 samples
# Select data subsets
# train_data = dataset["train"].select(range(0, 2000))  # Limit to 2000 samples
# eval_data = dataset["validation"].select(range(0, 100))  # Limit to 100 samples
# test_data = dataset["test"].select(range(0, 10))  # Limit to 10 samples

# Filter test data into intra-project and cross-project
intra_project_test = test_data.filter(lambda x: x["isa_cross_project_example"] == False)
cross_project_test = test_data.filter(lambda x: x["isa_cross_project_example"] == True)

# Load Phi-2 model and tokenizer
print("Loading Phi-2 model...")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
model.to(device)

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Retrieve related examples
def retrieve_related_examples(goal_type, train_data, top_k=3, max_tokens=400):
    """Retrieve semantically similar examples from training data"""
    try:
        goal_emb = embedder.encode([goal_type], convert_to_tensor=True, device=device)
        train_types = [ex["source_type"] for ex in train_data]
        train_emb = embedder.encode(train_types, convert_to_tensor=True, device=device)
        similarities = cosine_similarity(goal_emb.cpu().numpy(), train_emb.cpu().numpy())[0]
        top_indices = np.argsort(similarities)[::-1][:top_k].tolist()
        
        related_examples = []
        total_tokens = 0
        for idx in top_indices:
            example = train_data[int(idx)]
            ex_type = example["source_type"]
            ex_def = example["source_definition"]
            text = f"Example:\n{ex_type}\n{ex_def}"
            tokens = len(tokenizer(text)["input_ids"])
            if total_tokens + tokens <= max_tokens:
                related_examples.append(text)
                total_tokens += tokens
            else:
                break
        return related_examples
    except Exception as e:
        print(f"Error in retrieve_related_examples: {e}")
        return []

# Premise selection
def select_premises(goal_type, premises, top_k=5, max_tokens=300):
    """Select most relevant premises using semantic similarity"""
    if not premises:
        return []
    
    try:
        prem_emb = embedder.encode(premises, convert_to_tensor=True, device=device)
        goal_emb = embedder.encode([goal_type], convert_to_tensor=True, device=device)
        similarities = cosine_similarity(goal_emb.cpu().numpy(), prem_emb.cpu().numpy())[0]
        top_indices = np.argsort(similarities)[::-1][:top_k].tolist()
        
        selected = []
        total_tokens = 0
        for idx in top_indices:
            prem = premises[idx]
            tokens = len(tokenizer(prem)["input_ids"])
            if total_tokens + tokens <= max_tokens:
                selected.append(prem)
                total_tokens += tokens
            else:
                break
        return selected
    except Exception as e:
        print(f"Error in select_premises: {e}")
        return premises[:top_k]

# Create inference prompt
def prepare_inference_prompt(example):
    """Create inference prompt for RAG-based generation"""
    source_type = example["source_type"]
    file_context = example["file_context"]
    opens_and_abbrevs = example["opens_and_abbrevs"]
    ideal_premises = example["ideal_premises"]
    partial_definition = example["partial_definition"]
    
    opened_modules = "\n".join(
        [f"open {oa['full_module']}" if not oa["abbrev"] else f"module {oa['short_module']} = {oa['full_module']}"
         for oa in opens_and_abbrevs]
    )
    
    # Retrieve related examples
    related_examples = retrieve_related_examples(source_type, train_data)
    related_str = "\n\n".join(related_examples) if related_examples else "No related examples found."
    
    # Select premises
    selected_premises = select_premises(source_type, ideal_premises)
    premises_str = "\n".join([f"- {p}" for p in selected_premises])
    
    # Truncate context if too long
    context_tokens = len(tokenizer(file_context)["input_ids"])
    if context_tokens > 500:
        file_context = tokenizer.decode(tokenizer(file_context, max_length=500, truncation=True)["input_ids"])
    
    inference_prompt = f"""<|im_start|>system
You are an F* code synthesis assistant. Generate a COMPLETE F* definition (not just the type declaration) based on the given information. Always provide the full implementation.
<|im_end|>

<|im_start|>user
Type to implement:
```fstar
{source_type}
```

Opened modules:
```fstar
{opened_modules}
```

File context:
```fstar
{file_context}
```

Relevant premises:
```fstar
{premises_str}
```

Related examples from codebase:
```fstar
{related_str}
```

Partial definition to complete:
```fstar
{partial_definition}
```

Generate the COMPLETE F* definition (including implementation body). Provide only the complete definition:
<|im_end|>

<|im_start|>assistant
```fstar
"""
    
    return inference_prompt

# Inference function modified to generate 10 responses
def generate_fstar_code(example, max_new_tokens=512, num_responses=10):
    """Generate F* code using the pre-trained model with RAG, producing multiple responses"""
    inference_prompt = prepare_inference_prompt(example)
    
    inputs = tokenizer(inference_prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generated_codes = []
    with torch.no_grad():
        for _ in range(num_responses):
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            assistant_start = generated_text.find("<|im_start|>assistant")
            if assistant_start != -1:
                generated_part = generated_text[assistant_start:]
                fstar_match = re.search(r'```fstar\n(.*?)```', generated_part, re.DOTALL)
                if fstar_match:
                    generated_codes.append(fstar_match.group(1).strip())
                else:
                    fstar_start = generated_part.find('```fstar\n')
                    if fstar_start != -1:
                        code_part = generated_part[fstar_start + 9:]
                        end_match = re.search(r'```|\n<\|', code_part)
                        if end_match:
                            generated_codes.append(code_part[:end_match.start()].strip())
                        else:
                            generated_codes.append(code_part.strip())
                    else:
                        prompt_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
                        generated_codes.append(generated_text[prompt_length:].strip())
            else:
                prompt_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
                generated_codes.append(generated_text[prompt_length:].strip())
    
    return generated_codes

# Function to save test results in JSON format with multiple responses
def save_test_results_to_json(test_data, test_set_name, output_file):
    results = []
    for i, example in enumerate(test_data):
        print(f"Processing {test_set_name} example {i+1}/{len(test_data)}")
        try:
            generated_codes = generate_fstar_code(example, num_responses=10)
            result = dict(example)  # Copy original example data
            result["generated_response"] = {
                "responses": generated_codes,
                "is_match": any(code.strip() == example["source_definition"].strip() for code in generated_codes)
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing {test_set_name} example {i+1}: {e}")
            result = dict(example)
            result["generated_response"] = {
                "responses": [f"Error: {str(e)}"],
                "is_match": False
            }
            results.append(result)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

# Save intra-project and cross-project test results
print("Saving intra-project test results...")
save_test_results_to_json(intra_project_test, "Intra-Project", "intra_project_test_results.json")

print("Saving cross-project test results...")
save_test_results_to_json(cross_project_test, "Cross-Project", "cross_project_test_results.json")

# Print summary statistics
intra_results = []
with open("intra_project_test_results.json", 'r', encoding='utf-8') as f:
    intra_results = json.load(f)
cross_results = []
with open("cross_project_test_results.json", 'r', encoding='utf-8') as f:
    cross_results = json.load(f)

intra_matches = sum(1 for r in intra_results if r["generated_response"]["is_match"])
cross_matches = sum(1 for r in cross_results if r["generated_response"]["is_match"])
intra_total = len(intra_results)
cross_total = len(cross_results)

print(f"\nResults Summary:")
# print(f"Intra-Project: {intra_matches}/{intra_total} correct ({100.0 * intra_matches/intra_total:.2f}%)")
# print(f"Cross-Project: {cross_matches}/{cross_total} correct ({100.0 * cross_matches/cross_total:.2f}%)")

# Example usage for debugging
if len(train_data) > 0:
    print("\nExample generation for debugging:")
    example = train_data[0]
    generated_codes = generate_fstar_code(example, num_responses=10)
    print(f"Source type: {example['source_type']}")
    print(f"Generated responses:")
    for i, code in enumerate(generated_codes, 1):
        print(f"Response {i}: {code}")
    print(f"Ground truth: {example['source_definition']}")
    print(f"Match: {any(code.strip() == example['source_definition'].strip() for code in generated_codes)}")