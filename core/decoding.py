import torch
from core.model import softmax

def top_p_filter(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs - sorted_probs > top_p # [False False True]
    sorted_probs[mask] = 0.0
    probs_filtered = torch.zeros_like(probs)
    probs_filtered.scatter_(0, sorted_indices, sorted_probs)
    return probs_filtered / probs_filtered.sum()


def decode(model, prompt_tokens, max_new_tokens, eos_token_id, temperature=1.0, top_p=1.0, context_length=None):
    tokens = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = tokens if context_length is None else tokens[-context_length:]
            input_tensor = torch.tensor(input_ids).unsqueeze(0) # (1, seq_len)

            logits = model(input_tensor)[0, -1, :] # (vocab_size,) get last token
            logits = logits / temperature 
            probs = softmax(logits, -1)

            if top_p < 1.0:
                probs = top_p_filter(probs, top_p)

            # Sample
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

            if next_token == eos_token_id:
                break

    return tokens


