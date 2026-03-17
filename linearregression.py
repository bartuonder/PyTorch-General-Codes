import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt

df = pd.read_csv("06-study_hours_grades.csv")

x = torch.tensor(df["study_hours"].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(df["grade"].values, dtype=torch.float32).unsqueeze(1)

train_split = int(len(x) * 0.8)

x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.linear(a)

torch.manual_seed(42)
model = LinearRegression()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 250

for epoch in range(epochs):

    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")

print(model.state_dict())

plt.scatter(x_train, y_train, c="b", s=5, label="Train Data")
plt.scatter(x_test, y_test, c="g", s=5, label="Test Data")
plt.scatter(x_test, test_pred, c="r", s=5, label="Predictions")
plt.show()
