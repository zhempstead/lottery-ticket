import torch
import torch.nn as nn
import torch.optim as optim

import lt_model
from models.feedforward import SimpleFeedforward
from datasets.mnist import mnist_trainloader, mnist_testloader

LEARNING_RATE = 1.2e-3

def train_feedforward_mnist():
    
    model = SimpleFeedforward()
    train_loader = mnist_trainloader()
    test_loader = mnist_testloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def callback(epoch, model, running_loss):
        print(f'[{epoch + 1}] loss: {running_loss / 1000:.3f}')
        lt_model.test_classification(model, test_loader)
        torch.save(model.state_dict(), f"model/ff_mnist_{epoch + 1}.pt")

    model = lt_model.train(model, train_loader, criterion, optimizer, 10, callback) 
    #model.load_state_dict(torch.load("model/ff_mnist_10.pt"))

    for prune_amt in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"Pruning {prune_amt}")
        lt_model.prune_model(model, prune_amt, exclude_modules=['fc3'])
        # In the paper the final layer is pruned only half as much
        lt_model.prune_model(model, prune_amt/2, include_modules=['fc3'])
        lt_model.test_classification(model, test_loader)
        torch.save(model.state_dict(), f"model/ff_mnist_prune_{prune_amt}.pt")

if __name__ == '__main__':
    train_feedforward_mnist()
