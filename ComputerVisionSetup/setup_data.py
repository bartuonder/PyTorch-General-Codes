import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS,
):

    train_data = datasets.ImageFolder(root = train_dir, transform = transform, target_transform = None)
    test_data = datasets.ImageFolder(root = test_dir, transform = transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    return train_dataloader, test_dataloader, class_names