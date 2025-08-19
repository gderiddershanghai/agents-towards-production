# pip install torch transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Config
# -----------------------
@dataclass
class PPOConfig:
    model_name: str = "gpt2"
    max_new_tokens: int = 64
    batch_size: int = 4
    gen_temperature: float = 1.0
    kl_beta: float = 0.02            # KL control (policy vs reference)
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    lr_policy: float = 1e-5
    lr_value: float = 1e-4
    gamma: float = 1.0               # episodic, no discounting across tokens by default
    lam: float = 0.95                # GAE lambda
    pad_token_id: int = None

cfg = PPOConfig()

# -----------------------
# Policy / Reference / Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
cfg.pad_token_id = tokenizer.pad_token_id

policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
reference = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

# -----------------------
# Simple Value Head
# -----------------------
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, T, H], produce token-level values [B, T]
        values = self.value(hidden_states).squeeze(-1)
        # mask padding
        values = values * attention_mask
        return values

value_head = ValueHead(policy.config.n_embd).to(device)

opt_policy = AdamW(policy.parameters(), lr=cfg.lr_policy)
opt_value  = AdamW(value_head.parameters(), lr=cfg.lr_value)

# -----------------------
# Utility: compute token logprobs of provided continuations (teacher-forced scoring)
# -----------------------
def compute_logprobs(model, input_ids, attention_mask):
    """
    Returns per-token logprobs of the *labels* (next tokens) and logits.
    We score the continuation region; caller aligns labels accordingly.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits = outputs.logits  # [B, T, V]
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs, outputs.hidden_states[-1]  # last hidden for value head

def gather_logprobs(token_logprobs, labels):
    # token_logprobs: [B, T, V]; labels: [B, T] with -100 masked for ignored tokens
    B, T, V = token_logprobs.shape
    flat = token_logprobs.view(B*T, V)
    idx = labels.view(B*T, 1)
    picked = flat.gather(1, idx.clamp_min(0))  # gather needs non-negative; dummy for -100
    picked = picked.view(B, T)
    picked = torch.where(labels == -100, torch.zeros_like(picked), picked)
    return picked

# -----------------------
# Reward stub (replace with RM or human feedback)
# -----------------------
@torch.no_grad()
def reward_model_score(prompts, responses):
    """
    Return a scalar reward per sequence. Placeholder logic:
    +1 if the response contains the word 'therefore' else 0.
    Replace with a real reward model.
    """
    rewards = []
    for p, r in zip(prompts, responses):
        rewards.append(1.0 if "therefore" in r.lower() else 0.0)
    return torch.tensor(rewards, dtype=torch.float32, device=device)

# -----------------------
# KL penalty against reference (sequence-level, token-composed)
# -----------------------
@torch.no_grad()
def sequence_kl(policy_logprobs_tok, ref_logprobs_tok, mask):
    # D_KL(pi || ref) ≈ sum_t pi_tok * (log pi_tok - log ref_tok) is exact,
    # but we only know log-prob of chosen tokens. For PPO RLHF, a common proxy is:
    # KL ≈ sum_t (log pi(y_t) - log ref(y_t))
    kl_t = (policy_logprobs_tok - ref_logprobs_tok) * mask  # [B, T]
    kl_seq = kl_t.sum(dim=1)                                # [B]
    return kl_seq

# -----------------------
# GAE on sequence-level (broadcast per-token)
# For RLHF, rewards are sequence-level scalars; we broadcast them to response tokens.
# -----------------------
def compute_advantages(rewards, values_seq):
    # rewards: [B] sequence-level
    # values_seq: [B] baseline per sequence (mean of token values over response region)
    # A = R - V (episodic signal per sequence)
    advantages = rewards - values_seq
    returns = rewards  # episodic return
    return advantages, returns

# -----------------------
# Rollout (generation) helper
# -----------------------
@torch.no_grad()
def rollout_generate(prompts_ids, max_new_tokens, temperature):
    gen = policy.generate(
        input_ids=prompts_ids["input_ids"],
        attention_mask=prompts_ids["attention_mask"],
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return gen  # [B, prompt_len + gen_len]

def split_prompt_response(input_ids, prompt_lengths):
    # Returns masks for response region
    B, T = input_ids.shape
    mask = torch.zeros_like(input_ids)
    for i, L in enumerate(prompt_lengths):
        mask[i, L:] = 1
    return mask

# -----------------------
# One PPO iteration (single batch)
# -----------------------
def ppo_step(prompts_text):
    # 1) Tokenize prompts
    enc = tokenizer(prompts_text, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_lens = [l for l in enc["attention_mask"].sum(dim=1).tolist()]

    # 2) Generate responses with current policy (on-policy)
    gen = rollout_generate(enc, cfg.max_new_tokens, cfg.gen_temperature)
    full_attn = (gen != cfg.pad_token_id).long()
    resp_mask = split_prompt_response(gen, prompt_lens)  # 1 on response tokens

    # 3) Decode for reward model
    full_text = tokenizer.batch_decode(gen, skip_special_tokens=True)
    prompts_only = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)
    responses_only = []
    for txt, L in zip(full_text, prompt_lens):
        # not robust; in practice slice by tokens not strings. Kept simple here.
        responses_only.append(txt)  # entire decoded output as proxy
    rewards = reward_model_score(prompts_only, responses_only)  # [B]

    # 4) Teacher-forced scoring of the *generated* sequences under policy & reference
    with torch.no_grad():
        # old logprobs (behavior policy): score tokens under current policy BEFORE update
        # Prepare labels as next-token targets
        labels = gen.clone()
        labels[:, :1] = -100  # no next-token for first position
        policy_lp_tok_old, _ = compute_logprobs(policy, gen, full_attn)
        ref_lp_tok, _ = compute_logprobs(reference, gen, full_attn)
        lp_old = gather_logprobs(policy_lp_tok_old, labels)  # [B, T]
        lp_ref = gather_logprobs(ref_lp_tok, labels)         # [B, T]

    # Token mask for response region only
    mask = resp_mask * full_attn

    # Sequence-level KL penalty
    kl_seq = sequence_kl(lp_old, lp_ref, mask)               # [B]
    rewards_kl = rewards - cfg.kl_beta * kl_seq              # [B]

    # 5) Baseline (value) over response region
    with torch.no_grad():
        _, hs = compute_logprobs(policy, gen, full_attn)     # [B, T, H]
    values_tok = value_head(hs, full_attn)                   # [B, T]
    # average value over response tokens (avoid division by zero)
    denom = mask.sum(dim=1).clamp_min(1)
    values_seq = (values_tok * mask).sum(dim=1) / denom      # [B]

    # 6) Advantages and returns (sequence-level)
    advantages, returns = compute_advantages(rewards_kl, values_seq)  # [B], [B]
    # Broadcast to tokens in response region
    advantages_tok = (advantages.unsqueeze(1) * mask)                # [B, T]
    returns_tok    = (returns.unsqueeze(1)    * mask)                # [B, T]

    # 7) PPO policy update (multiple epochs over same batch)
    for _ in range(cfg.ppo_epochs):
        policy_lp_tok_new, _ = compute_logprobs(policy, gen, full_attn)
        lp_new = gather_logprobs(policy_lp_tok_new, labels)          # [B, T]
        # Ratio per token on response region
        ratio = torch.exp((lp_new - lp_old) * mask)                  # [B, T]
        # PPO clip
        unclipped = ratio * advantages_tok
        clipped   = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_tok
        ppo_loss_tok = -torch.minimum(unclipped, clipped)            # [B, T]
        # Mask out non-response tokens and average
        ppo_loss = (ppo_loss_tok.sum(dim=1) / denom).mean()

        opt_policy.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_policy.step()

    # 8) Value head regression to returns (MSE over response tokens)
    with torch.no_grad():
        _, hs = compute_logprobs(policy, gen, full_attn)
    values_tok = value_head(hs, full_attn)                            # [B, T]
    value_loss = ((values_tok - returns_tok)**2 * mask).sum(dim=1) / denom
    value_loss = value_loss.mean()

    opt_value.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
    opt_value.step()

    return {
        "mean_reward": rewards.mean().item(),
        "mean_kl": kl_seq.mean().item(),
        "ppo_loss": ppo_loss.item(),
        "value_loss": value_loss.item(),
    }

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    policy.train(); value_head.train()
    prompts = [
        "Explain why the sky is blue in one sentence.",
        "Write a polite refusal to share confidential information.",
        "Summarize the theory of evolution in 2 lines.",
        "Give a brief proof sketch that sqrt(2) is irrational."
    ]
    stats = ppo_step(prompts)
    print(stats)
