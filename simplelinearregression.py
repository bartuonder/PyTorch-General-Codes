import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt

df = pd.read_csv("06-study_hours_grades.csv")

x = torch.tensor(df["study_hours"].values)
y = torch.tensor(df["grade"].values)

train_split = int(len(x) * 0.8)

x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()

        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, a):
        return self.weights * a + self.bias


torch.manual_seed(42)
model_0 = SimpleLinearRegression()
#model_0 = torch.compile(model_0)

#print(list(model_0.parameters()))
#print(model_0.state_dict())
loss_fn = nn.MSELoss() #MSE
#loss_fn = nn.L1Loss() -> MAE
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) #lr=0.001

torch.manual_seed(42)
epochs = 200
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):

    model_0.train()

    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():

        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test) #loss_fn(test_pred, y_test.type(torch.float))

        if epoch % 5 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")


print(model_0.state_dict())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epoch_count, train_loss_values, label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.title("Eğitim ve Test Kayıp Grafiği")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_train, y_train, c="b", s=5, label="Train Data")
plt.scatter(x_test, y_test, c="g", s=5, label="Test Data")
plt.scatter(x_test, test_pred, c="r", s=5, label="Predictions")
plt.title("Model Tahmin Başarısı")
plt.xlabel("Study Hours")
plt.ylabel("Grade")
plt.legend()

plt.tight_layout()
plt.show()

