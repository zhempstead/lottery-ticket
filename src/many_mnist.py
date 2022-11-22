import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import lt_model
from models.feedforward import SimpleFeedforward
from datasets.mnist import mnist_trainloader, mnist_testloader

NUM_MODELS = 100
LEARNING_RATE = 1.2e-3
PRUNE_AMT = 0.75

def train(idx):
    model = SimpleFeedforward()
    train_loader = mnist_trainloader()
    test_loader = mnist_testloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f'Model {idx}...')

    def callback(epoch, model, running_loss):
        print(f'[{idx} : {epoch + 1}] loss: {running_loss / 1000:.3f}')

    model = lt_model.train(model, train_loader, criterion, optimizer, 10, callback) 

    torch.save(model.state_dict(), f"model/ff_mnist_unpruned_{idx}.pt")
    print(f"{idx}: Pruning {PRUNE_AMT}")
    lt_model.prune_model(model, PRUNE_AMT, exclude_modules=['fc3'])
    # In the paper the final layer is pruned only half as much
    lt_model.prune_model(model, PRUNE_AMT/2, include_modules=['fc3'])
    lt_model.test_classification(model, test_loader)
    torch.save(model.state_dict(), f"model/ff_mnist_prune_{PRUNE_AMT}_{idx}.pt")
    return idx

def analyze(idx):
    model = torch.load(f"model/ff_mnist_prune_0.75_{idx}.pt")
    mat_fc1 = model['fc1.weight'].numpy()
    mat_fc2 = model['fc2.weight'].numpy()
    mat_fc3 = model['fc3.weight'].numpy()

    w1_var, w1_out_cov, w1_out_corr = output_covariance(mat_fc1)
    n1_cov, n1_corr = node_covariance(mat_fc1, mat_fc2)
    w2_var, w2_in_cov, w2_in_corr = input_covariance(mat_fc2)
    _, w2_out_cov, w2_out_corr = output_covariance(mat_fc2)
    n2_cov, n2_corr = node_covariance(mat_fc2, mat_fc3)
    w3_var, w3_in_cov, w3_in_corr = input_covariance(mat_fc3)

    return (
        w1_var, w2_var, w3_var,
        w1_out_cov, w2_in_cov, w2_out_cov, w3_in_cov,
        w1_out_corr, w2_in_corr, w2_out_corr, w3_in_corr,
        n1_cov, n2_cov,
        n1_corr, n2_corr,
    )

def output_covariance(weights):
    '''
    Variance, covariance, and correlation of all pairs of weights that share output nodes
    Skips pairs if either weight is zero
    '''
    weights = np.abs(weights)
    left = np.repeat(weights, weights.shape[1])
    right = np.tile(weights, weights.shape[1]).flatten()
    stacked = np.stack([left, right], axis=1)
    stacked = stacked[~(stacked==0).any(1)]
    cov = np.cov(stacked[:, 0], stacked[:, 1])
    corr = np.corrcoef(stacked[:, 0], stacked[:, 1])[0,1]
    var = cov[0,0]
    cov = cov[0,1]
    return var, cov, corr

def input_covariance(weights):
    '''
    Variance, covariance, and correlation of all pairs of weights that share input nodes
    Skips pairs if either weight is zero
    '''
    return output_covariance(weights.T)

def node_covariance(input_weights, output_weights):
    '''
    Covariance and correlation of all pairs of input/output weights of nodes
    Skips pairs if either weight is zero
    '''
    assert input_weights.shape[0] == output_weights.shape[1]
    nn = input_weights.shape[0]

    input_weights = np.abs(input_weights)
    output_weights = np.abs(output_weights)

    left = np.repeat(input_weights, output_weights.shape[0])
    right = np.tile(output_weights.T, input_weights.shape[1]).flatten()
    stacked = np.stack([left, right], axis=1)
    stacked = stacked[~(stacked==0).any(1)]
    cov = np.cov(stacked[:, 0], stacked[:, 1])[0,1]
    corr = np.corrcoef(stacked[:, 0], stacked[:, 1])[0,1]
    return cov, corr


if __name__ == '__main__':
    outs = {metric: [] for metric in [
        'w1_var', 'w2_var', 'w3_var',
        'w1_out_cov', 'w2_in_cov', 'w2_out_cov', 'w3_out_cov',
        'w1_out_corr', 'w2_in_corr', 'w2_out_corr', 'w3_out_corr',
        'n1_cov', 'n2_cov',
        'n1_corr', 'n2_corr',
    ]}
    for idx in range(NUM_MODELS):
        # 'analyze' will analyze the saved models so if you've already run 'train' once you can
        # save a lot of time and comment out the call to 'train'
        train(idx)
        out = analyze(idx)
        for i, key in enumerate(outs.keys()):
            outs[key].append(out[i])
        print(f"Done with idx {idx}")
    pd.DataFrame(outs).to_csv('results', index=False)
