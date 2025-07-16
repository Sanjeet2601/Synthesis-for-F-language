import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import pandas as pd
import re
import os
import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import wandb
from tqdm import tqdm

# Initialize Weights & Biases
wandb.init(
    project="llama-with_own_premise_fstar-code-generation",
    name="llama-3-rag-finetuning",
    config={
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "embedder_model": "all-MiniLM-L6-v2",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "max_steps": 12000,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "warmup_steps": 100,
        "max_length": 2048,
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.8,
        "num_responses": 10,
        "rag_top_k": 3,
        "rag_max_tokens": 400,
        "inference_batch_size": 8
    }
)

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
train_data = dataset["train"]
eval_data = dataset["validation"]
test_data = dataset["test"] # Increased test size for better batching

# Log dataset statistics
wandb.log({
    "dataset/train_size": len(train_data),
    "dataset/eval_size": len(eval_data),
    "dataset/test_size": len(test_data)
})

# Filter test data
intra_project_test = test_data.filter(lambda x: x["isa_cross_project_example"] == False)
cross_project_test = test_data.filter(lambda x: x["isa_cross_project_example"] == True)

wandb.log({
    "dataset/intra_project_test_size": len(intra_project_test),
    "dataset/cross_project_test_size": len(cross_project_test)
})

# Cache training embeddings
print("Caching training embeddings...")
train_types = [ex["source_type"] for ex in train_data]
train_emb = embedder.encode(train_types, convert_to_tensor=True, device=device, batch_size=32)

# Load meta-llama/Llama-3.2-3B-Instruct model and tokenizer
print("Loading meta-llama/Llama-3.2-3B-Instruct model...")
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure LoRA for PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Log model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.log({
    "model/total_parameters": total_params,
    "model/trainable_parameters": trainable_params,
    "model/trainable_percentage": 100 * trainable_params / total_params
})

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
        return tuple(related_examples)
    except Exception as e:
        print(f"Error in retrieve_related_examples: {e}")
        return ()

# Prompt creation functions
def prepare_training_prompt(example):
    """Create training prompt with ground truth answer using Llama chat template"""
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
    
    context_tokens = len(tokenizer(file_context)["input_ids"])
    if context_tokens > 500:
        file_context = tokenizer.decode(tokenizer(file_context, max_length=500, truncation=True)["input_ids"])
    
    messages = [
        {
            "role": "system",
            "content": "You are an F* code synthesis assistant. Generate a COMPLETE F* definition based on the given information."
        },
        {
            "role": "user",
            "content": f"""Type to implement:
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
{ideal_premises}
```

Related examples:
```fstar
{related_str}
```

Partial definition:
```fstar
{partial_definition}
```

Generate the COMPLETE F* definition:"""
        },
        {
            "role": "assistant",
            "content": f"```fstar\n{example['source_definition']}\n```"
        }
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def prepare_inference_prompt(example):
    """Create inference prompt without ground truth answer using Llama chat template"""
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
    
    context_tokens = len(tokenizer(file_context)["input_ids"])
    if context_tokens > 500:
        file_context = tokenizer.decode(tokenizer(file_context, max_length=500, truncation=True)["input_ids"])
    
    messages = [
        {
            "role": "system",
            "content": "You are an F* code synthesis assistant. Generate a COMPLETE F* definition based on the given information."
        },
        {
            "role": "user",
            "content": f"""Type to implement:
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
{ideal_premises}
```

Related examples:
```fstar
{related_str}
```

Partial definition:
```fstar
{partial_definition}
```

Generate the COMPLETE F* definition:"""
        }
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

# Batch preprocessing for inference
def preprocess_batch_for_inference(batch):
    """Preprocess batch of examples for inference"""
    texts = [prepare_inference_prompt(example) for example in batch]
    
    tokenized = tokenizer(
        texts,
        max_length=1536,  # Shorter for inference to allow generation space
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    return tokenized

# Extract F* code from generated text
def extract_fstar_code(generated_text):
    """Extract F* code from generated text"""
    if '```fstar' in generated_text:
        fstar_start = generated_text.rfind('```fstar') + 8
        code_part = generated_text[fstar_start:]
        end_pos = code_part.find('```')
        code = code_part[:end_pos].strip() if end_pos != -1 else code_part.strip()
    else:
        # Fallback: extract everything after the last assistant message
        assistant_start = generated_text.rfind('<|start_header_id|>assistant<|end_header_id|>')
        if assistant_start != -1:
            code = generated_text[assistant_start + 47:].strip()
        else:
            code = generated_text.strip()
    
    return code

# Optimized batch inference function
def generate_fstar_code_batch(dataset, batch_size=8, max_new_tokens=512, num_responses=10):
    """Generate F* code for a batch of examples with progress tracking"""
    results = []
    
    # Convert dataset to list if it's not already
    if hasattr(dataset, '__iter__') and not isinstance(dataset, list):
        dataset = list(dataset)
    
    # Create progress bar
    progress_bar = tqdm(
        range(0, len(dataset), batch_size), 
        desc="Batch inference", 
        unit="batch"
    )
    
    for i in progress_bar:
        batch = dataset[i:i+batch_size]
        
        try:
            # Preprocess batch
            tokenized = preprocess_batch_for_inference(batch)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            # Repeat inputs for multiple responses per example
            batch_size_actual = input_ids.size(0)
            repeated_input_ids = input_ids.repeat(num_responses, 1)
            repeated_attention_mask = attention_mask.repeat(num_responses, 1)
            
            # Generate responses
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=repeated_input_ids,
                    attention_mask=repeated_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
                
                # Decode generated texts
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract F* code from each response
                all_codes = [extract_fstar_code(text) for text in generated_texts]
                
                # Group codes by example
                codes_by_example = [
                    all_codes[j:j+num_responses] 
                    for j in range(0, len(all_codes), num_responses)
                ]
                
        except Exception as e:
            print(f"Error in batch generation: {e}")
            # Create empty responses for failed batch
            codes_by_example = [[""] * num_responses for _ in range(len(batch))]
        
        # Process results for each example in the batch
        for j, generated_codes in enumerate(codes_by_example):
            if i + j < len(dataset):
                example = batch[j]
                result = dict(example)
                result["generated_response"] = {
                    "responses": generated_codes,
                    "is_match": any(
                        code.strip() == example["source_definition"].strip() 
                        for code in generated_codes if code
                    )
                }
                results.append(result)
        
        # Update progress bar with current accuracy
        if results:
            current_matches = sum(1 for r in results if r["generated_response"]["is_match"])
            current_accuracy = 100.0 * current_matches / len(results)
            progress_bar.set_postfix({
                'accuracy': f'{current_accuracy:.1f}%',
                'matches': f'{current_matches}/{len(results)}'
            })
    
    return results

# Optimized test results saving function
def save_test_results_optimized(dataset, test_set_name, output_file, batch_size=8):
    """Save test results with optimized batch processing"""
    print(f"Processing {test_set_name} test set with {len(dataset)} examples...")
    
    if len(dataset) == 0:
        print(f"Skipping {test_set_name} test set: no examples.")
        wandb.log({
            f"evaluation/{test_set_name.lower()}_inference_time_seconds": 0,
            f"evaluation/{test_set_name.lower()}_inference_time_per_example": 0,
            f"evaluation/{test_set_name.lower()}_accuracy": 0
        })
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return 0, 0, 0
    
    start_time = time.time()
    results = generate_fstar_code_batch(dataset, batch_size=batch_size, num_responses=10)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    correct_predictions = sum(1 for r in results if r["generated_response"]["is_match"])
    total_predictions = len(results)
    accuracy = 100.0 * correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Log metrics to wandb
    wandb.log({
        f"evaluation/{test_set_name.lower()}_inference_time_seconds": inference_time,
        f"evaluation/{test_set_name.lower()}_inference_time_per_example": inference_time / len(dataset),
        f"evaluation/{test_set_name.lower()}_accuracy": accuracy,
        f"evaluation/{test_set_name.lower()}_correct_predictions": correct_predictions,
        f"evaluation/{test_set_name.lower()}_total_predictions": total_predictions
    })
    
    print(f"Generated responses in {inference_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    # Save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")
    
    return accuracy, correct_predictions, total_predictions

# Apply preprocessing (for training)
print("Preprocessing training data...")
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names, batch_size=100)
print("Preprocessing evaluation data...")
tokenized_eval = eval_data.map(preprocess_function, batched=True, remove_columns=eval_data.column_names, batch_size=100)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama3_rag_fstar_finetuned",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    max_steps=12000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    optim="adamw_torch_fused",
    report_to="wandb",
    gradient_accumulation_steps=4,
    warmup_steps=100,
    save_total_limit=3,
    run_name="llama-3-rag-fstar-finetuning",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
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
model.save_pretrained("./llama3_rag_fstar_finetuned")
tokenizer.save_pretrained("./llama3_rag_fstar_finetuned")

# Save model as wandb artifact
artifact = wandb.Artifact("llama3-rag-fstar-model", type="model")
artifact.add_dir("./llama3_rag_fstar_finetuned")
wandb.log_artifact(artifact)

# OPTIMIZED BATCH INFERENCE SECTION
print("\n" + "="*60)
print("STARTING OPTIMIZED BATCH INFERENCE")
print("="*60)

# Run optimized inference on test sets
print("Running optimized inference on test sets...")
intra_accuracy, intra_correct, intra_total = save_test_results_optimized(
    intra_project_test, "Intra-Project", "intra_project_test_results.json", batch_size=8
)

cross_accuracy, cross_correct, cross_total = save_test_results_optimized(
    cross_project_test, "Cross-Project", "cross_project_test_results.json", batch_size=8
)

# Calculate and log overall results
overall_accuracy = 100.0 * (intra_correct + cross_correct) / (intra_total + cross_total) if (intra_total + cross_total) > 0 else 0

wandb.log({
    "results/intra_project_accuracy": intra_accuracy,
    "results/cross_project_accuracy": cross_accuracy,
    "results/overall_accuracy": overall_accuracy,
    "results/intra_project_correct": intra_correct,
    "results/intra_project_total": intra_total,
    "results/cross_project_correct": cross_correct,
    "results/cross_project_total": cross_total,
    "results/overall_correct": intra_correct + cross_correct,
    "results/overall_total": intra_total + cross_total
})

# Create and log results summary table
results_table = wandb.Table(columns=["Test Set", "Accuracy (%)", "Correct", "Total"])
results_table.add_data("Intra-Project", intra_accuracy, intra_correct, intra_total)
results_table.add_data("Cross-Project", cross_accuracy, cross_correct, cross_total)
results_table.add_data("Overall", overall_accuracy, intra_correct + cross_correct, intra_total + cross_total)
wandb.log({"results/summary_table": results_table})

print(f"\nOPTIMIZED INFERENCE RESULTS SUMMARY:")
print(f"Intra-Project: {intra_correct}/{intra_total} ({intra_accuracy:.2f}%)")
print(f"Cross-Project: {cross_correct}/{cross_total} ({cross_accuracy:.2f}%)")
print(f"Overall: {intra_correct + cross_correct}/{intra_total + cross_total} ({overall_accuracy:.2f}%)")

# Example generation for debugging
if len(train_data) > 0:
    print("\n" + "="*60)
    print("EXAMPLE GENERATION FOR DEBUGGING")
    print("="*60)
    
    example = train_data[0]
    print(f"Processing example: {example['source_type']}")
    
    # Generate using batch function for consistency
    results = generate_fstar_code_batch([example], batch_size=1, num_responses=5)
    if results:
        generated_codes = results[0]["generated_response"]["responses"]
        is_match = results[0]["generated_response"]["is_match"]
        
        print(f"\nSource type: {example['source_type']}")
        print(f"Ground truth:\n{example['source_definition']}")
        print(f"\nGenerated responses:")
        for i, code in enumerate(generated_codes, 1):
            print(f"\nResponse {i}:\n{code}")
            print(f"Match: {code.strip() == example['source_definition'].strip()}")
        
        print(f"\nOverall match: {is_match}")
        
        # Log example to wandb
        wandb.log({
            "examples/debug_example_match": is_match,
            "examples/debug_source_type": example['source_type'],
            "examples/debug_ground_truth": example['source_definition'],
            "examples/debug_generated_1": generated_codes[0] if len(generated_codes) > 0 else "",
            "examples/debug_generated_2": generated_codes[1] if len(generated_codes) > 1 else ""
        })

# Save example results as artifact
if len(train_data) > 0:
    example_results = {
        "source_type": example['source_type'],
        "ground_truth": example['source_definition'],
        "generated_responses": generated_codes,
        "match": is_match
    }
    
    with open("example_generation.json", "w") as f:
        json.dump(example_results, f, indent=2)
    
    example_artifact = wandb.Artifact("example-generation", type="predictions")
    example_artifact.add_file("example_generation.json")
    wandb.log_artifact(example_artifact)

# Finish wandb run
wandb.finish()

print("\nExperiment completed successfully!")
print("✓ Training completed")
print("✓ Batch inference completed with progress tracking")
print("✓ Results saved and logged to W&B")
print(f"✓ Check your W&B dashboard: llama-with_own_premise_fstar-code-generation")