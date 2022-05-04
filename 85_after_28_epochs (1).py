import torch  # idk which imports are needed in this barebones version
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 128, 3, 1, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=2)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, 1)
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1)

        self.maxPool = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.2)
        self.drop4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        i = 0
        f = lambda i: [i + 1, print(i, x.shape)][0]  # i = f(i) will print the shape of x

        x = nn.functional.rrelu(self.conv1(x))
        x = self.drop1(x)
        x = self.maxPool(x)
        x = nn.functional.rrelu(self.conv2(x))
        x = self.drop2(x)
        x = self.maxPool(x)
        x = nn.functional.rrelu(self.conv3(x))
        x = nn.functional.rrelu(self.conv4(x))
        x = self.drop3(x)
        x = self.maxPool(x)
        x = nn.functional.rrelu(self.conv5(x))
        x = nn.functional.rrelu(self.conv6(x))
        x = self.drop4(x)
        x = nn.functional.rrelu(self.conv7(x))
        # i=f(i)-*-
        # i=f(i)
        x=x.reshape(x.shape[0], -1)
       # print(x.shape)
        x = nn.functional.rrelu(self.fc1(x))
        x = nn.functional.rrelu(self.fc2(x))
        x = self.fc3(x)
        # i=f(i)
        return x


if __name__ == "__main__":
    if torch.cuda.is_available():  # use gpu if possible
        device = torch.device("cuda")
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

        print("Test accuracy was", 100 * float(correct) / float(samples))
        print()


    epochs = 40
    batch_size = 16
    learning_rate = 0.0003

    train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1,
                                                                             hue=0.1),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])



    # get training/test data from CIFAR10 dataset
    train_data = torchvision.datasets.CIFAR10(root="./dataset",
                                              train=True,
                                              transform=train_transform,
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(root="./dataset",
                                             train=False,
                                             transform=test_transform,
                                             download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.0015)

    #optimizer = optimizer.Adam([{ 'params': transfer_model.layer4.parameters(), 'lr': found_lr /3}, { 'params': transfer_model.layer3.parameters(), 'lr': found_lr /9},], lr=found_lr)

    test_per_epoch = True
    train_for_time = 0  # how many minutes to train for (and then finish current epoch)

    if train_for_time:
        epochs = train_for_time * 1000
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

        print("Epoch", epoch + 1, "complete")
        print("Loss was", epoch_loss / len(train_loader))
        print()
        if test_per_epoch:
            test()

        if train_for_time and time.time() - start >= train_for_time * 60:
            break

    if not test_per_epoch:
        test()




