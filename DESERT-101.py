import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
import os
from PIL import Image
import random
from torchinfo import summary
from torch.nn import ReLU
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-> Şu an kullanılan cihaz: {device}")

data_path = Path("data/")
image_path = data_path / "desert101"

#def check_data(dir_path):
#    for dirpath, dirnames, filenames in os.walk(dir_path):
#        print(f"# of directories: {len(dirnames)} and {len(filenames)} images in {dirpath}")

#check_data(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

#random.seed(42)
#image_path_list = list(image_path.glob("*/*/*.jpg"))
#random_image = random.choice(image_path_list)
#img = Image.open(random_image)
#plt.imshow(img)

data_transform = transforms.Compose([
    transforms.Resize(size = (64, 64)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])



class DesertClassifier(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))



#summary(model = model_0, input_size = [1, 3, 64, 64])

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):

    model.train()

    train_loss = 0
    train_accuracy = 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_accuracy += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)

    return train_loss, train_accuracy


def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):

    model.eval()

    test_loss = 0
    test_accuracy = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_accuracy += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)

    return test_loss, test_accuracy

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 10):

    results = {"train_loss": [],
               "train_accuracy": [],
               "test_loss": [],
               "test_accuracy": []
               }

    for epoch in range(epochs):

        train_loss, train_accuracy = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        test_loss, test_accuracy = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)

        print(f"Epochs: {epoch}, Train Loss: {train_loss}, "
              f"Train Accuracy: {train_accuracy} Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_accuracy"].append(train_accuracy.item() if isinstance(train_accuracy, torch.Tensor) else train_accuracy)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_accuracy"].append(test_accuracy.item() if isinstance(test_accuracy, torch.Tensor) else test_accuracy)

    return results


def plot_loss_curves(results):

    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_accuracy"]
    test_accuracy = results["test_accuracy"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.plot(epochs, test_loss, "r", label="Testing Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, "b", label="Training Accuracy")
    plt.plot(epochs, test_accuracy, "r", label="Testing Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


if __name__ == "__main__":

    train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform, target_transform=None)

    class_names = train_data.classes

    BATCH_SIZE = 32
    NUM_WORKERS = 0

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True)

    model_0 = DesertClassifier(input_shape=3, hidden_units=32, output_shape=len(class_names)).to(device)


    EPOCHS = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model_0.parameters(), lr = 0.001)

    model_0_results = train(model = model_0,
                          train_dataloader = train_dataloader,
                          test_dataloader = test_dataloader,
                          optimizer = optimizer,
                          loss_fn = loss_fn,
                          epochs = EPOCHS)


    online_image_path = data_path / "baklava.jpg"

    single_image = torchvision.io.read_image(str(online_image_path)).type(torch.float32)
    single_image /= 255.0

    #plt.imshow(single_image.permute(1, 2, 0))
    #plt.title(single_image.shape)

    single_image_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
    ])

    single_image = single_image_transform(single_image)

    single_image = single_image.unsqueeze(dim = 0)

    single_image = single_image.to(device)

    model_0.eval()
    with torch.inference_mode():
        logits = model_0(single_image)
        probs = torch.softmax(logits, dim = 1)
        pred_idx = probs.argmax(dim = 1).item()

    print("Predicted label: ", class_names[pred_idx])

    plot_loss_curves(model_0_results)













