import torch
import torchvision
import torchvision.transforms as transforms

TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

BATCH_SIZE = 60

def mnist_trainloader():
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=TRANSFORM)
    return torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

def mnist_testloader():
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=TRANSFORM)
    return torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
