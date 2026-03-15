import torch
from einops import rearrange, einsum
import math
import numpy as np

def cross_entropy(output: torch.Tensor, targets: torch.Tensor):
    output = output - torch.max(output, dim=-1, keepdim=True).values # (batch, seq_len, vocab_size)
    exp_x = torch.exp(output)
    logsoftmax = output - torch.log(torch.sum(exp_x, dim=-1, keepdim=True)) # (batch, seq_len, vocab_size)
    targets = targets.unsqueeze(-1) # (batch, seq_len, 1)
    output_logits = torch.gather(logsoftmax, dim=-1, index=targets) # (batch, seq_len, 1)
    return -output_logits.squeeze(-1).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            beta1, beta2 = betas
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0
                
                state["t"] += 1
                t = state["t"]
                grad = p.grad.data
                state["m"] = beta1*state["m"] + (1-beta1)*grad
                state["v"] = beta2*state["v"] + (1-beta2)*grad**2
                state["lr"] = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                p.data -= state["lr"] * state["m"] / (torch.sqrt(state["v"]+eps))
                p.data -= lr*weight_decay*p.data

def learning_rate_schedule(t, a_max, a_min, t_w, t_c):
    if t < t_w:
        return t*a_max / t_w
    elif t <= t_c:
        return a_min + 0.5 * (1 + math.cos((t-t_w)*math.pi/(t_c-t_w))) * (a_max-a_min)
    else:
        return a_min        

def gradient_clipping(params: list, max_l2: float, eps: float=1e-6):
    total_norm = math.sqrt(sum(p.grad.norm()**2 for p in params if p.grad is not None))
    if total_norm > max_l2:
        scale = max_l2 / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    
def data_loading(x, batch_size, context_length, device):
    starts = torch.randint(0, len(x)-context_length, (batch_size,)) # ex [1, 3, 6]
    inputs = torch.stack([torch.tensor(x[i:i+context_length], dtype=torch.long) for i in starts]) # (batch_size, context_length)
    targets = torch.stack([torch.tensor(x[i+1:i+context_length+1], dtype=torch.long) for i in starts])
    return (inputs.to(device), targets.to(device))

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]







