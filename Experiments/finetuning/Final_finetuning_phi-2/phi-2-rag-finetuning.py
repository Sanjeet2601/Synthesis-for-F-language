import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import pandas as pd
import re
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Specify GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
# train_data = dataset["train"].select(range(0, 2000))  # Limit to 2000 samples
# eval_data = dataset["validation"].select(range(0, 100))  # Limit to 100 samples
# test_data = dataset["test"].select(range(0, 10))  # Limit to 10 samples
# # Select data subsets
train_data = dataset["train"]  # Use full train dataset
eval_data = dataset["validation"]
test_data = dataset["test"]

# Filter test data
intra_project_test = test_data.filter(lambda x: x["isa_cross_project_example"] == False)
cross_project_test = test_data.filter(lambda x: x["isa_cross_project_example"] == True)

# Cache training embeddings
print("Caching training embeddings...")
train_types = [ex["source_type"] for ex in train_data]
train_emb = embedder.encode(train_types, convert_to_tensor=True, device=device, batch_size=32)

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

# Configure LoRA for PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)

# Retrieve related examples
@lru_cache(maxsize=1000)
def retrieve_related_examples(goal_type, top_k=3, max_tokens=400):
    """Retrieve semantically similar examples from training data with caching"""
    try:
        goal_emb = embedder.encode([goal_type], convert_to_tensor=True, device=device)
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
        return tuple(related_examples)  # Return tuple for cache compatibility
    except Exception as e:
        print(f"Error in retrieve_related_examples: {e}")
        return ()

# Premise selection
def select_premises(goal_type, premises, top_k=5, max_tokens=300):
    """Select most relevant premises using semantic similarity"""
    if not premises:
        return []
    
    try:
        prem_emb = embedder.encode(premises, convert_to_tensor=True, device=device, batch_size=32)
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

# Prompt creation functions
def prepare_training_prompt(example):
    """Create training prompt with ground truth answer"""
    source_type = example["source_type"]
    file_context = example["file_context"]
    opens_and_abbrevs = example["opens_and_abbrevs"]
    ideal_premises = example["ideal_premises"]
    partial_definition = example["partial_definition"]
    
    opened_modules = "\n".join(
        [f"open {oa['full_module']}" if not oa["abbrev"] else f"module {oa['short_module']} = {oa['full_module']}"
         for oa in opens_and_abbrevs]
    )
    
    related_examples = retrieve_related_examples(source_type)
    related_str = "\n\n".join(related_examples) if related_examples else "No related examples found."
    
    selected_premises = select_premises(source_type, ideal_premises)
    premises_str = "\n".join([f"- {p}" for p in selected_premises])
    
    context_tokens = len(tokenizer(file_context)["input_ids"])
    if context_tokens > 500:
        file_context = tokenizer.decode(tokenizer(file_context, max_length=500, truncation=True)["input_ids"])
    
    return f"""<|im_start|>system
You are an F* code synthesis assistant. Generate a COMPLETE F* definition based on the given information.
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

Related examples:
```fstar
{related_str}
```

Partial definition:
```fstar
{partial_definition}
```

Generate the COMPLETE F* definition:
<|im_end|>

<|im_start|>assistant
```fstar
{example["source_definition"]}
```<|im_end|>"""

def prepare_inference_prompt(example):
    """Create inference prompt without ground truth answer"""
    source_type = example["source_type"]
    file_context = example["file_context"]
    opens_and_abbrevs = example["opens_and_abbrevs"]
    ideal_premises = example["ideal_premises"]
    partial_definition = example["partial_definition"]
    
    opened_modules = "\n".join(
        [f"open {oa['full_module']}" if not oa["abbrev"] else f"module {oa['short_module']} = {oa['full_module']}"
         for oa in opens_and_abbrevs]
    )
    
    related_examples = retrieve_related_examples(source_type)
    related_str = "\n\n".join(related_examples) if related_examples else "No related examples found."
    
    selected_premises = select_premises(source_type, ideal_premises)
    premises_str = "\n".join([f"- {p}" for p in selected_premises])
    
    context_tokens = len(tokenizer(file_context)["input_ids"])
    if context_tokens > 500:
        file_context = tokenizer.decode(tokenizer(file_context, max_length=500, truncation=True)["input_ids"])
    
    return f"""<|im_start|>system
You are an F* code synthesis assistant. Generate a COMPLETE F* definition based on the given information.
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

Related examples:
```fstar
{related_str}
```

Partial definition:
```fstar
{partial_definition}
```

Generate the COMPLETE F* definition:
<|im_end|>

<|im_start|>assistant
```fstar
"""

# Preprocessing for training
def preprocess_function(examples):
    """Preprocess examples for training"""
    texts = [prepare_training_prompt({k: examples[k][i] for k in examples.keys()}) for i in range(len(examples["source_type"]))]
    
    tokenized = tokenizer(
        texts,
        max_length=2048,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    tokenized["labels"][tokenized["labels"] == tokenizer.pad_token_id] = -100
    
    return tokenized

# Apply preprocessing
print("Preprocessing training data...")
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names, batch_size=100)
print("Preprocessing evaluation data...")
tokenized_eval = eval_data.map(preprocess_function, batched=True, remove_columns=eval_data.column_names, batch_size=100)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./phi2_rag_fstar_finetuned",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=1e-4,
    per_device_train_batch_size=2,  # Increased batch size
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    max_steps=10000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    optim="adamw_torch_fused",  # Optimized AdamW
    report_to="none",
    gradient_accumulation_steps=2,  # Reduced to balance memory
    warmup_steps=100,
    save_total_limit=3,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving fine-tuned model...")
model.save_pretrained("./phi2_rag_fstar_finetuned")
tokenizer.save_pretrained("./phi2_rag_fstar_finetuned")

# Inference function for multiple responses
def generate_fstar_code(example, max_new_tokens=512, num_responses=10):
    """Generate multiple F* code responses"""
    inference_prompt = prepare_inference_prompt(example)
    inputs = tokenizer(inference_prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    
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
            fstar_start = generated_text.rfind('```fstar\n') + 9
            if fstar_start > 8:
                code_part = generated_text[fstar_start:]
                end_pos = code_part.find('```')
                if end_pos == -1:
                    end_pos = code_part.find('\n<|')
                code = code_part[:end_pos].strip() if end_pos != -1 else code_part.strip()
            else:
                code = generated_text[generated_text.rfind('<|im_start|>assistant')+22:].strip()
            
            generated_codes.append(code)
    
    return generated_codes

# Save test results to JSON
def save_test_results_to_json(test_data, test_set_name, output_file):
    results = []
    for i, example in enumerate(test_data):
        print(f"Processing {test_set_name} example {i+1}/{len(test_data)}")
        try:
            generated_codes = generate_fstar_code(example, num_responses=10)
            result = dict(example)
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
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

# Save test results
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