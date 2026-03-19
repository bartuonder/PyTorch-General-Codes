import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #label encoding
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.utilities.plot import plot_confusion_matrix

df = pd.read_csv("09-iris.csv")

df.drop("Id", axis=1, inplace=True)

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
y = df["Species"].values

le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class IrisClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )


    def forward(self, x):
        return self.linear_layer_stack(x)



model = IrisClassificationModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

accuracy = MulticlassAccuracy(num_classes=3)

epochs = 200

train_loss_values = []
test_loss_values = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):

    model.train()

    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    pred = torch.softmax(logits, dim=1).argmax(dim=1)
    acc = accuracy(pred, y_train).item() * 100

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    model.eval()
    with torch.inference_mode():

        test_logits = model(X_test)
        test_loss = loss_fn(test_logits, y_test)

        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_acc = accuracy(test_pred, y_test).item() * 100


    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")


cm = MulticlassConfusionMatrix(num_classes=3)

matrix = cm(test_pred, y_test)

print(matrix)

plot_confusion_matrix(matrix)
plt.show()