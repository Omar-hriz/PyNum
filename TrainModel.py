import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transform
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

bach_size = 128
input_size = 28*28
target_size = 10
step = 10**(-3)

dataset = MNIST(root="data/", train=True, transform=transform.ToTensor())
train_dataset, validation_dataset = random_split(dataset, [50000, 10000])
test_dataset = MNIST(root="data/", train=False, transform=transform.ToTensor())
train_data_loader = DataLoader(train_dataset, bach_size, shuffle=True)
validation_data_loader = DataLoader(validation_dataset, bach_size)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item()/len(labels)

def predictImage(img,model):
    xi = img.unsqueeze(0)
    ot = model(xi)
    _, preds = torch.max(ot, dim=1)
    return preds[0].item()

class MyModule(nn.Module):
    def __init__(self):
        super(). __init__()
        self.linear = nn.Linear(input_size, target_size)

    def forward(self, xi):
        return self.linear(xi.reshape(-1, input_size))



if __name__ == "__main__":
    model = MyModule()
    opt = torch.optim.SGD(model.parameters(), lr=step)
    x = []
    for epoch in range(20):
        for images, labels in train_data_loader:
            outputs = model(images)
            # probs = F.softmax(outputs, dim=1)
            # max_prob, index = torch.max(probs, dim=1)
            # print(accuracy(outputs, labels))
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
        metric = 0
        avg_loss = 0
        for images, labels in train_data_loader:
            outputs = model(images)
            avg_loss += F.cross_entropy(outputs, labels)
            metric += accuracy(outputs, labels)
        print("{} / 100".format(epoch))
        x.append(metric/len(train_data_loader))
        print(metric/len(train_data_loader))
        print(avg_loss/len(train_data_loader))

#enregiter le modele a la fin
#torch.save(model.state_dict(), "Model.pth")
