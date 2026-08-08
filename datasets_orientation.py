import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import cv2
from PIL import Image

batchsize = 16

train_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.6),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

predict_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def get_train_loader():
    train_dataset = datasets.ImageFolder(
        root = 'C:/Users/Robyn/Desktop/Research Project/rotated_dataset/train',
        transform = train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batchsize,
        shuffle = True,
        num_workers = 0
    )

    return train_loader

def get_test_loader():
    test_dataset = datasets.ImageFolder(
        root='C:/Users/Robyn/Desktop/Research Project/rotated_dataset/test',
        transform = test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batchsize,
        shuffle = False,
        num_workers = 0
    )

    return test_loader

def get_prediction_tensor(card):
   

    card_rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)

    card_pil = Image.fromarray(card_rgb)

    card_to_tensor = predict_transform(card_pil)

    return card_to_tensor.unsqueeze(0)
