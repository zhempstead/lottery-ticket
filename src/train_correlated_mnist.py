import copy

import pandas as pd
import torch.nn as nn
import torch.optim as optim

import lt_model
from models.feedforward import SimpleFeedforward
from datasets.mnist import mnist_trainloader, mnist_testloader

LEARNING_RATE = 1.2e-3

#CORRELATE_FACTORS = [1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
CORRELATE_FACTORS = [1.0, 1.1]

def train_correlated_mnist():
    
    model = SimpleFeedforward()
    train_loader = mnist_trainloader()
    test_loader = mnist_testloader()

    criterion = nn.CrossEntropyLoss()

    def callback(epoch, model, running_loss):
        if epoch >= 0:
            print(f'[{epoch + 1}] loss: {running_loss / 1000:.3f}')

    #print("Baseline:")
    #bmodel = copy.deepcopy(model)
    #optimizer = optim.Adam(bmodel.parameters(), lr=LEARNING_RATE)
    #bmodel = lt_model.train(bmodel, train_loader, criterion, optimizer, 1, callback, mod=4)
    #lt_model.test_classification(bmodel, test_loader)

    accuracies = []
    for correlate_factor in CORRELATE_FACTORS:
        print(f"Correlate factor {correlate_factor}:")
        cmodel = copy.deepcopy(model)
        cmodel.correlate_nodes(correlate_factor) 
        optimizer = optim.Adam(cmodel.parameters(), lr=LEARNING_RATE)
        cmodel = lt_model.train(cmodel, train_loader, criterion, optimizer, 1, callback, mod=4)
        accuracies.append(lt_model.test_classification(cmodel, test_loader))
    return pd.DataFrame({'correlate_factor': CORRELATE_FACTORS, 'test_accuracy': accuracies})

if __name__ == '__main__':
    dfs = []
    for idx in range(50):
        df = train_correlated_mnist()
        df['idx'] = idx
        dfs.append(df)
        full = pd.concat(dfs)
        full.to_csv('full_correlate.csv', index=False)
