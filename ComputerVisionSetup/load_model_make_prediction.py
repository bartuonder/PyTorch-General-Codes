from torchvision import transforms
from model_creation import DesertClassifier
import torch
from pathlib import Path
import torchvision
import setup_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "data/desert101/train"
test_dir = "data/desert101/test"

_, _, class_names = setup_data.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]),
    batch_size=32
)

MODEL_SAVE_PATH = "models/desert_classifier.pth"

loaded_model = DesertClassifier(
    input_shape=3,
    hidden_units=32,
    output_shape=len(class_names)
)

loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
loaded_model.to(device)

data_path = Path("data/")
online_image_path = data_path / "baklava.jpg"

single_image = torchvision.io.read_image(str(online_image_path)).type(torch.float32)
single_image /= 255.0

single_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.Normalize(mean=[0.5483, 0.4638, 0.3865],
                         std=[0.2173, 0.2279, 0.2263])
])

single_image = single_image_transform(single_image)
single_image = single_image.unsqueeze(dim = 0).to(device)

loaded_model.eval()
with torch.inference_mode():
    logits = loaded_model(single_image)
    probs = torch.softmax(logits, dim = 1)
    pred_idx = probs.argmax(dim = 1).item()

print("Predicted label: ", class_names[pred_idx])