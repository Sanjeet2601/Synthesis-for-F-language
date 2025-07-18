# Synthesis-for-F-language
Implementation of paper Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming: F* Code Generation with Fine-Tuned Phi-2 and Llama-3 Models

This repository contains the implementation for fine-tuning two large language models, Phi-2 and Llama-3.2-3B-Instruct, for generating F* code, a functional programming language used for formal verification. The project leverages Retrieval-Augmented Generation (RAG) to enhance code generation by retrieving semantically similar examples from the training data. 
# Fine-Tuning: 
Fine-tuned Phi-2 and Llama-3 models on the microsoft/FStarDataSet dataset using LoRA.
# RAG Integration: 
Retrieves relevant F* code examples using the all-MiniLM-L6-v2 sentence transformer for improved context.
# Prompting Techniques: 
Uses structured prompts with system instructions, file context, opened modules, premises, and partial definitions.
Prompt: 
<img width="560" height="311" alt="image" src="https://github.com/user-attachments/assets/9470c49f-5e46-4ae2-87f8-6a12fe1eb83f" />

<img width="625" height="429" alt="image" src="https://github.com/user-attachments/assets/6258f0c5-a58e-4308-89d2-75ddfcbc7bc1" />

# Access Requirements:
Hugging Face account and access to meta-llama/Llama-3.2-3B-Instruct (requires approval).
Access to the microsoft/FStarDataSet dataset.


# Install Dependencies:
pip install torch transformers datasets peft sentence-transformers scikit-learn wandb tqdm

# Dataset
The project uses the microsoft/FStarDataSet dataset, which contains F* code examples with fields like source_type, source_definition, file_context, opens_and_abbrevs, ideal_premises, and partial_definition. The dataset is split into:

Training: Used for fine-tuning.
Validation: Used for evaluating during training.
Test: Split into intra-project and cross-project subsets for final evaluation.
# Result on phi-2 and Llama-3.2
<img width="1630" height="1027" alt="image" src="https://github.com/user-attachments/assets/ccd24e10-f1ea-4933-afa0-b4892a5a5a53" />

