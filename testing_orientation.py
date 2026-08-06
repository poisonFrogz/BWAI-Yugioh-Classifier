import torch
import cnnmodel
import datasets_orientation as datasets

num_classes = 2
if __name__ == '__main__':
    model = cnnmodel.CnnModel(num_classes)
    test_loader = datasets.get_test_loader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/orientation_model.pth'
    model.load_state_dict(torch.load(path))

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy: {}%'.format(100 * correct / total))