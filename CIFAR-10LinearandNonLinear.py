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


class CIFAR10Classifier(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


torch.manual_seed(42)
model = CIFAR10Classifier(input_shape=3072, hidden_units=128, output_shape=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def calculate_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100


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

model_results = evaluate_model_performance(model=model, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_function=calculate_accuracy)

class CIFAR10ClassifierNonLinear(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model1 = CIFAR10ClassifierNonLinear(input_shape=3072, hidden_units=128, output_shape=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)

epochs = 10

for epoch in range(epochs):
    train_loss = 0
    model1.train()

    for batch, (X, y) in enumerate(train_dataloader):

        X, y = X.to(device), y.to(device)

        y_pred = model1(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    test_loss, test_accuracy = 0, 0

    model1.eval()

    with torch.inference_mode():

        for X, y in test_dataloader:

            X, y = X.to(device), y.to(device)

            test_pred = model1(X)
            test_loss += loss_fn(test_pred, y).item()
            test_accuracy += calculate_accuracy(y, test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)

    print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")

model1_results = evaluate_model_performance(model=model1, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_function=calculate_accuracy)

print(model_results)
print(model1_results)



















