import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-> Şu an kullanılan cihaz: {device}")

data_path = Path("data/")
image_path = data_path / "desert101"

train_dir = image_path / "train"
test_dir = image_path / "test"


train_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class DesertClassifier(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_units * 32 * 32, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):

    model.train()
    train_loss, train_accuracy = 0, 0
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

    return train_loss / len(dataloader), train_accuracy / len(dataloader)


def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_accuracy = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_accuracy += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    return test_loss / len(dataloader), test_accuracy / len(dataloader)



def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          scheduler, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 10):

    results = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(epochs):
        train_loss, train_accuracy = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn,
                                                optimizer=optimizer)
        test_loss, test_accuracy = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch: {epoch + 1}/{epochs} | LR: {current_lr:.5f} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)
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

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

    class_names = train_data.classes

    BATCH_SIZE = 16
    NUM_WORKERS = 0

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)


    model_0 = DesertClassifier(input_shape=3, hidden_units=32, output_shape=len(class_names)).to(device)

    EPOCHS = 50
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_fn=loss_fn,
                            epochs=EPOCHS)


    if (data_path / "baklava.jpg").exists():
        online_image_path = data_path / "baklava.jpg"
        single_image = torchvision.io.read_image(str(online_image_path)).type(torch.float32)
        single_image /= 255.0

        single_image_transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        single_image = single_image_transform(single_image)
        single_image = single_image.unsqueeze(dim=0).to(device)

        model_0.eval()
        with torch.inference_mode():
            logits = model_0(single_image)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()

        print("\nPredicted label: ", class_names[pred_idx])

    plot_loss_curves(model_0_results)