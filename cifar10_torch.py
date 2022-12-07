import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

transformer = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)

def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
    image = image.clip(0, 1)
    return image

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

dataiter = iter(training_loader)
images, labels = dataiter.next()

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
    # row 2 column 10
    ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[i]))
    ax.set_title(classes[labels[i].item()])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 12
running_loss_history = []
running_correct_history = []
validation_running_loss_history = []
validation_running_correct_history = []

for e in range(epochs):

    running_loss = 0.0
    running_correct = 0.0
    validation_running_loss = 0.0
    validation_running_correct = 0.0

    for inputs, labels in training_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_correct += torch.sum(preds == labels.data)
        running_loss += loss.item()



    else:
    # 훈련팔 필요가 없으므로 메모리 절약
        with torch.no_grad():
            for val_input, val_label in validation_loader:
                val_input = val_input.to(device)
                val_label = val_label.to(device)
                val_outputs = model(val_input)
                val_loss = criterion(val_outputs, val_label)

                _, val_preds = torch.max(val_outputs, 1)
                validation_running_loss += val_loss.item()
                validation_running_correct += torch.sum(val_preds == val_label.data)


        epoch_loss = running_loss / len(training_loader)
        epoch_acc = running_correct.float() / len(training_loader)
        running_loss_history.append(epoch_loss)
        running_correct_history.append(epoch_acc)

        val_epoch_loss = validation_running_loss / len(validation_loader)
        val_epoch_acc = validation_running_correct.float() / len(validation_loader)
        validation_running_loss_history.append(val_epoch_loss)
        validation_running_correct_history.append(val_epoch_acc)

        print("===================================================")
        print("epoch: ", e + 1)
        print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
        print("validation loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
    # RGB세개 1채널, 20개 특징 추출, filter 크기, stride 1
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
    # 전에서 20개
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
    # 0.5 가 권장 할 만하대
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # flatten
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x