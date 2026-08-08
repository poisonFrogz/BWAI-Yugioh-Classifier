import torch
import torch.nn as nn
import resnet_model
import datasets_type as datasets
import time


num_classes = 9
learning_rate = 0.001
num_epochs = 4

if __name__ == '__main__':

    train_loader = datasets.get_train_loader()
    test_loader = datasets.get_test_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = resnet_model.CnnModel(num_classes)
    model = model.to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)


    best_accuracy = 0.0
    print(next(model.parameters()).device)
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()

        running_loss = 0.0
        # Load data batches with dataloader
        for i,(images,labels) in enumerate(train_loader):
            # Move tensors to GPU or CPU
            images = images.to(device)
            labels = labels.to(device)

            #Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backward Propagation and Optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

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

            test_accuracy = 100 * correct / total


        end_time = time.time()
        mins,secs = divmod(end_time-start_time,60)


        print('Epoch [{},{}], Loss: {:.4f} ¦ Test Accuracy: {:.2f}% ¦ Time: {}m {:.2f}s'.format(epoch+1, num_epochs, average_loss,test_accuracy,int(mins),secs))

        if test_accuracy > best_accuracy:
            print(f'Accuracy improved from {best_accuracy:.2f}% to {test_accuracy:.2f}%.')
            best_accuracy = test_accuracy
            path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/card_type_model.pth'
            torch.save(model.state_dict(), path)

