import os
import torch
import urllib.request


SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"
DATA_PATH = "data/shakespeare.txt"


def download_shakespeare():
    if os.path.exists(DATA_PATH):
        #print("Data file exists already.")
        return

    os.makedirs(name = "data", exist_ok = True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)


class CharacterTokenizer:

    def __init__(self, text: str):
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        self.char_to_id = {}

        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index

        self.id_to_char = {index: char for index, char in enumerate(self.characters)}

        #print(f"Vocab Size: {self.vocab_size}")
        #print(f"Characters: {''.join(self.characters)}")
        #print(self.char_to_id)

    def encode(self, text: str) -> list:
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])
        return ids

    def decode(self, ids: list) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        characters = []
        for id in ids:
            characters.append(self.id_to_char[id])
        return ''.join(characters)



def load_data(train_split: float = 0.9):

    download_shakespeare()

    with open(DATA_PATH, "r", encoding = "utf-8") as file:
        text = file.read()

    tokenizer = CharacterTokenizer(text)

    #print(tokenizer.encode("bartu"))
    #print(tokenizer.decode([40, 39, 56, 58, 59]))

    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype = torch.long)

    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]

    #print(f"Train Size: {len(train_data)}")
    #print(f"Test Size: {len(test_data)}")

    return train_data, test_data, tokenizer



def get_batch(data: torch.Tensor, block_size: int, batch_size: int):

    max_start = len(data) - block_size - 1
    positions = torch.randint(max_start, (batch_size, ))
    #print(f"Positions: {positions}")

    x_list = []
    y_list = []

    for pos in positions:
        x_list.append(data[pos : pos + block_size])
        y_list.append(data[pos + 1 : pos + block_size + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    #print(x)
    #print(y)

    return x, y



if __name__ == "__main__":

    train_data, test_data, tokenizer = load_data()
    x, y = get_batch(train_data, 256, 5)
