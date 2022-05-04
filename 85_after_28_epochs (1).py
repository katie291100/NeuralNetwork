import torch # idk which imports are needed in this barebones version
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature
import matplotlib.pyplot as plt
import numpy as np
import time

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(3, 128, 3, 1)
    self.bn1 = nn.BatchNorm2d(128)

    self.conv2 = nn.Conv2d(128, 256, 3, 1)
    self.bn2 = nn.BatchNorm2d(256)

    self.conv3 = nn.Conv2d(256, 256, 3, 1, padding="same")
    self.bn3 = nn.BatchNorm2d(100)

    self.conv4 = nn.Conv2d(256, 512, 3, 1)
    self.conv5 = nn.Conv2d(512, 512, 3, 1, padding="same")
    self.conv6 = nn.Conv2d(512, 512, 3, 1)

    self.maxPool = nn.MaxPool2d(2)

    self.drop1 = nn.Dropout(0.3)
    self.drop2 = nn.Dropout(0.4)
    self.drop3 = nn.Dropout(0.5)
    self.drop4 = nn.Dropout(0.6)
    self.drop5 = nn.Dropout(0.7)

    self.fc1 = nn.Linear(18432, 8192)
    self.fc2 = nn.Linear(8192, 8192)
    self.fc3 = nn.Linear(8192, 10)

  def forward(self, x):
    #i = 0
    #f = lambda i: [i+1,print(i,x.shape)][0]

    x = nn.functional.rrelu(self.conv1(x))
    x = nn.functional.rrelu(self.conv2(x))
    x = self.drop1(x)

    x = self.maxPool(x)
    skip = x.clone()

    x = nn.functional.rrelu(self.conv3(x))
    x = self.drop2(x)

    x = x + skip

    x = nn.functional.rrelu(self.conv4(x))
    x = self.maxPool(x)
    x = self.drop3(x)

    skip = x.clone()

    x = nn.functional.rrelu(self.conv5(x))
    x = x + skip
    x = self.drop4(x)
    x = nn.functional.rrelu(self.conv6(x))
    x = self.maxPool(x)
    x = self.drop5(x)

    x = x.reshape(x.shape[0], -1)
    #i=f(i)
    x = nn.functional.rrelu(self.fc1(x))

    x = nn.functional.rrelu(self.fc2(x))
    x = self.fc3(x)
    #i=f(i)
    return x


if __name__ == "__main__":
    if torch.cuda.is_available(): # use gpu if possible
        device = torch.device("cuda")
        print("cuda)")
    else:
        device = torch.device("cpu")

    model = CNN().to(device)

    def test():
        print("Testing")
        with torch.no_grad():
            correct = 0
            samples = 0

        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = outputs.max(1)

            samples += labels.size(0)
            correct += (predictions == labels).sum()

        print("Test accuracy was",100*float(correct)/float(samples))
        print()


    epochs = 400
    batch_size = 12
    learning_rate = 0.0001

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
    ])

    # get training/test data from CIFAR10 dataset
    train_data = torchvision.datasets.CIFAR10(root = "./dataset",
                                              train = True,
                                              transform = transform,
                                              download = True)
    test_data = torchvision.datasets.CIFAR10(root = "./dataset",
                                             train = False,
                                             transform = transform,
                                             download = True)
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                              batch_size = batch_size,
                                              shuffle = False,
                                              num_workers = 4)


    found_lr = 1e-2

    # try other loss functions/optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
      { 'params': model.conv4.parameters(), 'lr': found_lr /3},
      { 'params': model.conv3.parameters(), 'lr': found_lr /9},
      ], lr=found_lr)

    unfreeze_layers = [model.conv3, model.conv4]
    for layer in unfreeze_layers:
        for param in layer.parameters():
            param.requires_grad = True
    test_per_epoch = True
    train_for_time = 0 # how many minutes to train for (and then finish current epoch)
    if train_for_time:
      epochs = train_for_time*1000
    start = time.time()

    for epoch in range(epochs):
        print("Training")
        epoch_loss = 0
        previous_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forwards
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("Epoch", epoch+1, "complete")
        print("Loss was", epoch_loss/len(train_loader))
        print()
        if test_per_epoch:
            test()

        if train_for_time and time.time()-start >= train_for_time*60:
            break

    if not test_per_epoch:
        test()
