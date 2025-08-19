# minimal_sft.py
# pip install "transformers>=4.41" datasets accelerate bitsandbytes peft
import os, json, random
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# -----------------------------
# 1) Config
# -----------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")  # swap for a larger causal LM (e.g., "mistralai/Mistral-7B-v0.1")
DATA_PATH  = os.environ.get("DATA_PATH", "sft_data.jsonl")  # local JSONL: {"instruction": "...", "input": "...", "response": "..."}
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./sft-out")
MAX_LEN    = int(os.environ.get("MAX_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
LR         = float(os.environ.get("LR", "2e-5"))
EPOCHS     = int(os.environ.get("EPOCHS", "1"))
GRAD_ACC   = int(os.environ.get("GRAD_ACC", "8"))
WARMUP     = int(os.environ.get("WARMUP", "100"))
SEED       = int(os.environ.get("SEED", "42"))

random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 2) Load / build dataset
# -----------------------------
# If you have a local JSONL, use "json" loader pointing to a list of files.
# Each line should look like:
# {"instruction": "Summarize:", "input": "Long passage...", "response": "Short summary..."}

if not os.path.exists(DATA_PATH):
    # Create a tiny toy dataset if none exists (for demo)
    with open(DATA_PATH, "w") as f:
        print(json.dumps({"instruction": "Write a haiku about the ocean.", "input": "", "response": "Waves whisper softly\nMoonlight dances on the tide\nBlue depths keep their tales"}), file=f)
        print(json.dumps({"instruction": "Turn this into SQL", "input": "List all users who signed up in 2024.", "response": "SELECT * FROM users WHERE signup_date >= '2024-01-01' AND signup_date < '2025-01-01';"}), file=f)
        print(json.dumps({"instruction": "Explain simply", "input": "What is overfitting?", "response": "When a model memorizes training data patterns and fails to generalize to new data."}), file=f)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=SEED)  # small eval split
train_ds, eval_ds = dataset["train"], dataset["test"]

# -----------------------------
# 3) Tokenizer & prompt template
# -----------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token  # for causal models

# A simple instruction-tuning style template
def format_example(example: Dict[str, str]) -> str:
    instruction = example.get("instruction", "").strip()
    user_input  = example.get("input", "").strip()
    response    = example.get("response", "").strip()

    if user_input:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{user_input}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
    return prompt, response

# -----------------------------
# 4) Tokenization with label masking
# -----------------------------
def tokenize_with_labels(example: Dict[str, str]) -> Dict[str, List[int]]:
    prompt, response = format_example(example)

    # Tokenize prompt and response separately
    prompt_ids   = tok(prompt, add_special_tokens=False).input_ids
    response_ids = tok(response + tok.eos_token, add_special_tokens=False).input_ids

    # Concatenate for model input
    input_ids = prompt_ids + response_ids
    input_ids = input_ids[:MAX_LEN]

    # Labels: -100 for prompt tokens (ignored by loss), response tokens get actual ids
    labels = [-100] * len(prompt_ids) + response_ids
    labels = labels[:MAX_LEN]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

train_tok = train_ds.map(tokenize_with_labels, remove_columns=train_ds.column_names, desc="Tokenizing train")
eval_tok  = eval_ds.map(tokenize_with_labels,  remove_columns=eval_ds.column_names,  desc="Tokenizing eval")

# Basic dynamic padding collator for causal LM with masked labels
@dataclass
class CausalDataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # Pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Labels need manual padding (pad with -100)
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for f in features:
            labels = f["labels"]
            pad_len = max_len - len(labels)
            padded_labels.append(labels + [-100] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

collator = CausalDataCollator(tok)

# -----------------------------
# 5) Model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Optional: enable gradient checkpointing for large models
model.gradient_checkpointing_enable()

# -----------------------------
# 6) Training setup
# -----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=WARMUP,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),  # use bf16 on modern GPUs, else ignored
    fp16=not torch.cuda.is_available() and False,  # set True if you want fp16 on GPU that supports it
    report_to="none",
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)

# -----------------------------
# 7) Quick sanity check: generate
# -----------------------------
model.eval()
prompt = "### Instruction:\nExplain simply\n\n### Input:\nWhat is overfitting?\n\n### Response:\n"
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tok.eos_token_id,
    )
print(tok.decode(out[0], skip_special_tokens=True))
