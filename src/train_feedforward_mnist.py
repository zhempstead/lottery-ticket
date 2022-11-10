import torch
import torch.nn as nn
import torch.optim as optim

from models.feedforward import SimpleFeedforward
from datasets.mnist import mnist_trainloader, mnist_testloader

EPOCHS = 50
LEARNING_RATE = 1.2e-3

def train(output):
    model = SimpleFeedforward()
    loader = mnist_trainloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    torch.save(model.state_dict(), Path(output))



if __name__ == '__main__':
    train("./foo")
