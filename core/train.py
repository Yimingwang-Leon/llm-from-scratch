import torch
import numpy as np
import argparse
import time
import json
import os
from core.model import TransformerLM
from core.training import cross_entropy, AdamW, learning_rate_schedule, gradient_clipping, data_loading, save_checkpoint, load_checkpoint


class ExperimentLogger:
    def __init__(self, log_path: str, config: dict):
        self.log_path = log_path
        self.start_time = time.time()
        with open(log_path, "w") as f:
            f.write(json.dumps({"type": "config", **config}) + "\n")

    def log(self, step: int, **kwargs):
        record = {"type": "metric", "step": step, "wallclock": time.time() - self.start_time, **kwargs}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    logger = ExperimentLogger(
        log_path=os.path.join(args.out_dir, f"{args.run_name}.jsonl"),
        config=vars(args)
    )

    # load data
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    # initilize model
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        context_length=args.context_length,
        theta=args.theta
    ).to(device)

    # initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    start_step = 0
    if args.checkpoint:
        start_step = load_checkpoint(args.checkpoint, model, optimizer)

    # training loops
    for step in range(start_step, args.num_steps):
        # update lr
        lr = learning_rate_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        loss.backward()
        gradient_clipping(list(model.parameters()), args.max_grad_norm)
        optimizer.step()

        # Logging
        if step % args.log_interval == 0:
            print(f"step {step}, train loss: {loss.item():.4f}, lr: {lr:.6f}")
            logger.log(step, train_loss=loss.item(), lr=lr)

        # Validation
        if step % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = data_loading(val_data, args.batch_size, args.context_length, device)
                val_logits = model(val_inputs)
                val_loss = cross_entropy(val_logits, val_targets)
            model.train()
            print(f"step {step}, val loss: {val_loss.item():.4f}")
            logger.log(step, val_loss=val_loss.item())

        # Checkpoint
        if step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, f"{args.out_dir}/ckpt_{step}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--cosine_steps", type=int, default=10000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    args = parser.parse_args()
    main(args)
