import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
batch_size = 16

train_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomRotation((0,180)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.6),
    transforms.Normalize(
        mean = [0.5,0.5,0.5],
        std = [0.5,0.5,0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5,0.5,0.5],
        std = [0.5,0.5,0.5]
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


