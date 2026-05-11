import os
import time
import torch
from model import GPT
from dataset import load_data, get_batch



EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
BLOCK_SIZE = 256
DROPOUT = 0.1

BATCH_SIZE = 64
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
WARMUP_ITERS = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"



def get_learning_rate(iteration):
    if iteration < WARMUP_ITERS:
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    else:
        return LEARNING_RATE


@torch.no_grad()
def evaluate(model, train_data, test_data):
    model.eval()

    results = {}

    for name, data in [("train", train_data), ("test", test_data)]:
        total_loss = 0.0

        for _ in range(100):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item()

        results[name] = total_loss / 100

    model.train()
    return results


@torch.no_grad()
def generate_sample(model, tokenizer):
    model.eval()

    prompt = "ROMEO: "

    prompt_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(prompt_ids, dtype = torch.long, device = DEVICE).unsqueeze(0)

    output_ids = model.generate(input_ids, max_new_tokens = 200, temperature = 0.8)

    model.train()

    return tokenizer.decode(output_ids[0])


def train():
    train_data, test_data, tokenizer = load_data(train_split = 0.9)
    vocab_size = tokenizer.vocab_size

    model = GPT(vocab_size = vocab_size,
                 embedding_dim = EMBEDDING_DIM,
                 num_heads = NUM_HEADS,
                 dropout = DROPOUT,
                 block_size = BLOCK_SIZE,
                 num_layers = NUM_LAYERS
                )

    model = model.to(device = DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    os.makedirs("checkpoints", exist_ok = True)

    start_time = time.time()

    for iteration in range(MAX_ITERS):
        lr = get_learning_rate(iteration)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)

        x, y = x.to(device = DEVICE), y.to(device = DEVICE)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        optimizer.step()

        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = evaluate(model, train_data, test_data)
            elapsed = time.time() - start_time

            print(f"Iteration: {iteration}, "
                  f"Train Loss: {losses["train"]}, "
                  f"Test Loss: {losses["test"]}, "
                  f"Learning Rate: {lr}, "
                  f"Time: {elapsed}")

            if iteration > 0:
                print(generate_sample(model, tokenizer))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "iteration": MAX_ITERS,
        "test_loss": losses["test"],
        "config": {
            "vocab_size": vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "block_size": BLOCK_SIZE
        }
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    total_time = time.time() - start_time
    print(f"Total Time: {total_time}")
    print(f"Final Test Loss: {losses["test"]}")
    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()