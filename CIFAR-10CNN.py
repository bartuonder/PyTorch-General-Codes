import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

train_data = datasets.CIFAR10(root="data",
                              train=True,
                              download=True,
                              transform=transform)

test_data = datasets.CIFAR10(root="data",
                             train=False,
                             download=True,
                             transform=transform)

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


def calculate_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100

def evaluate_model_performance(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_function):

    loss = 0
    accuracy = 0

    model.eval()

    with torch.inference_mode():

        for X, y in data_loader:

            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss += loss_fn(y_pred, y).item()
            accuracy += accuracy_function(y, y_pred.argmax(dim=1))

        loss /= len(data_loader)
        accuracy /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss,
            "model_accuracy": accuracy,
            }

class CIFAR10CNN(torch.nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 8 * 8, out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.block_2(self.block_1(x)))


model = CIFAR10CNN(input_shape=3, hidden_units=32, output_shape=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    train_loss = 0
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):

        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    test_loss, test_accuracy = 0, 0

    model.eval()

    with torch.inference_mode():

        for X, y in test_dataloader:

            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()
            test_accuracy += calculate_accuracy(y, test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)

    print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")

model_results = evaluate_model_performance(model=model, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_function=calculate_accuracy)
print(model_results)