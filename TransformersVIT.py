import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

IMG_SIZE =  224

def to_RGB(img):
    return img.convert("RGB")

train_tf = transforms.Compose([
    transforms.Lambda(to_RGB),
    transforms.RandomResizedCrop(IMG_SIZE, scale = (0.6, 1.0), ratio = (0.75, 1.33)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p = 0.25, scale = (0.02, 0.15), ratio = (0.3, 3.3), value = "random")
])

test_tf = transforms.Compose([
    transforms.Lambda(to_RGB),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

#data_path = Path("data/")
#image_path = data_path / "Caltech101"
test_ratio = 0.2
#train_dir = image_path / "train"
#test_dir = image_path / "test"

ds_train_full = datasets.Caltech101(root="data",
                              download=True,
                              transform=train_tf)

ds_test_full = datasets.Caltech101(root="data",
                              download=True,
                              transform=test_tf)

n_total = len(ds_train_full)
n_test = int(n_total * test_ratio)
n_train = n_total - n_test

g = torch.Generator().manual_seed(42)
perm = torch.randperm(n_total, generator = g).tolist()

train_idx = perm[:n_train]
test_idx = perm[n_train:]

train_ds = Subset(ds_train_full, train_idx)
test_ds = Subset(ds_test_full, test_idx)

BATCH_SIZE = 32

train_dataloader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False)

class_names = ds_train_full.categories

height = 224
width = 224
color_channels = 3
patch_size = 16
number_of_patches = int((height * width) / (patch_size ** 2))

embedding_layer_input_shape = (height, width, color_channels)
embedding_layer_output_shape = (number_of_patches, patch_size ** 2 * color_channels)

print(embedding_layer_input_shape)
print(embedding_layer_output_shape)


class PatchEmbedding(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 embedding_dim: int = 768,
                 patch_size: int = 16):

        super().__init__()

        self.patcher = nn.Conv2d(in_channels = in_channels,
                                 out_channels = embedding_dim,
                                 kernel_size = patch_size,
                                 stride = patch_size,
                                 padding = 0)

        self.flatten = nn.Flatten(start_dim = 2, end_dim = 3)

    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)


class MultiheadSelfAttentionBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=False)
        return attn_output

class MLPBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    def forward(self, x):

        x =  self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

class Vit(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 embedding_dim: int = 768,
                 patch_size: int = 16,
                 img_size: int = 224,
                 num_transformer_layer: int = 12,
                 num_heads: int = 12,
                 attn_dropout: float = 0.0,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 num_classes: int = 101):

        super().__init__()

        self.num_patches = int((img_size * img_size) / (patch_size ** 2))

        self.class_embedding = nn.Parameter(torch.rand(1, 1, embedding_dim), requires_grad = True)

        self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, embedding_dim), requires_grad = True)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential( * [ TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                           num_heads=num_heads,
                                                           attn_dropout=attn_dropout,
                                                           mlp_size=mlp_size,
                                                           mlp_dropout=mlp_dropout) for _ in range(num_transformer_layer)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):

        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim = 1)

        x = self.position_embedding + x

        x = self.transformer_encoder(x)

        x = self.classifier(x[:,0])

        return x

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results

vit = Vit().to(device=device)

optimizer = torch.optim.Adam(params=vit.parameters(), lr=3e-3, weight_decay=0.3)

loss_fn = torch.nn.CrossEntropyLoss()

results = train(model=vit,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epochs=10)




















