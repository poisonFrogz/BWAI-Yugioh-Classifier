import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import cv2
from PIL import Image

batch_size = 16

train_transform = transforms.Compose([
    transforms.Pad(padding=20, fill=(255,255,255), padding_mode='constant'),
    transforms.Resize((371,256)),
    transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.98, 1.02)),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Pad(padding=20, fill=(255,255,255), padding_mode='constant'),
    transforms.Resize((371,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

predict_transform = transforms.Compose([
    transforms.Pad(padding=20, fill=(255,255,255), padding_mode='constant'),
    transforms.Resize((371,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_train_loader ():
    train_dataset = datasets.ImageFolder(
        root = 'C:/Users/Robyn/Desktop/Research Project/cards_split/train',
        transform = train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
        drop_last = True
    )

    return train_loader

def get_test_loader ():
    test_dataset = datasets.ImageFolder(
        root='C:/Users/Robyn/Desktop/Research Project/cards_split/test',
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return test_loader

def get_prediction_tensor(card):
    card_to_predict = card

    card_rgb = cv2.cvtColor(card_to_predict, cv2.COLOR_BGR2RGB)

    card_pil = Image.fromarray(card_rgb)

    card_to_tensor = predict_transform(card_pil)

    return card_to_tensor.unsqueeze(0)


