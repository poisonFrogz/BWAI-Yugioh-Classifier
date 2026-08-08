import torch
import torch.nn as nn

import orientation_model
import resnet_model
import datasets_orientation as datasets
import time

num_classes = 2
learning_rate = 0.001
num_epochs = 10

if __name__ == '__main__':

    train_loader = datasets.get_train_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = orientation_model.CnnModel(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)

    model.train()
    print(next(model.parameters()).device)

    for epoch in range(num_epochs):
        start_time = time.time()

        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Move to GPU
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

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)
        average_loss = running_loss / len(train_loader)

        print('Epoch [{},{}], Loss: {:.4f} ¦ Time: {}m {:.2f}s'.format(epoch + 1, num_epochs, average_loss, mins, secs))

    path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/orientation_model.pth'
    torch.save(model.state_dict(), path)
