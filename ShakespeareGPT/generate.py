import torch
from model import GPT
from dataset import download_shakespeare, CharacterTokenizer, DATA_PATH


def load_model(checkpoint_path: str = "checkpoints/model.pt"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)

    config = checkpoint["config"]

    download_shakespeare()

    with open(DATA_PATH, "r", encoding = "utf-8") as file:
        text = file.read()

    tokenizer = CharacterTokenizer(text)

    model = GPT(
        vocab_size = config["vocab_size"],
        embedding_dim = config["embedding_dim"],
        num_heads = config["num_heads"],
        num_layers = config["num_layers"],
        block_size = config["block_size"],
        dropout = 0.0
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device = device)
    model.eval()

    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, max_token: int = 500, temperature: float = 0.8):

    prompt_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(prompt_ids, dtype = torch.long, device = device).unsqueeze(0)

    output_ids = model.generate(input_ids, max_new_tokens = max_token, temperature = temperature)

    generated_text = tokenizer.decode(output_ids[0])

    return generated_text


if __name__ == "__main__":
    model, tokenizer, device = load_model()

    while True:
        try:
            prompt = input("\nYour Prompt: ")
            if prompt.lower() in ["quit", "q", "exit"]:
                print("Farewell!")
                break
            if not prompt:
                continue

            generated_text = generate(
                model = model,
                tokenizer = tokenizer,
                device = device,
                prompt = prompt,
                max_token = 500,
                temperature = 0.8
            )

            print("\n" + generated_text)

        except KeyboardInterrupt:
            print("Farewell!")
            break