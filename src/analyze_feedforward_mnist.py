import matplotlib.pyplot as plt
import pandas as pd
import torch

import lt_model
from models.feedforward import SimpleFeedforward
from datasets.mnist import mnist_trainloader, mnist_testloader

PRUNE_AMT = 0.75

def analyze_feedforward_mnist():
    model = torch.load(f"model/ff_mnist_prune_{PRUNE_AMT}.pt")
    l1_nonzero_input = (model['fc1.weight'] != 0).sum(axis=1)
    l1_nonzero_output = (model['fc2.weight'] != 0).sum(axis=0)
    l2_nonzero_input = (model['fc2.weight'] != 0).sum(axis=1)
    l2_nonzero_output = (model['fc3.weight'] != 0).sum(axis=0)

    l1_positive_input = (model['fc1.weight'] > 0).sum(axis=1) / l1_nonzero_input
    l1_positive_output = (model['fc2.weight'] > 0).sum(axis=0) / l1_nonzero_output
    l2_positive_input = (model['fc2.weight'] > 0).sum(axis=1) / l2_nonzero_input
    l2_positive_output = (model['fc3.weight'] > 0).sum(axis=0) / l2_nonzero_output

    l1_nz_avg_input = (model['fc1.weight'].sum(axis=1)) / l1_nonzero_input
    l1_nz_avg_output = (model['fc2.weight'].sum(axis=0)) / l1_nonzero_output
    l2_nz_avg_input = (model['fc2.weight'].sum(axis=1)) / l2_nonzero_input
    l2_nz_avg_output = (model['fc3.weight'].sum(axis=0)) / l2_nonzero_output

    l1_nz_abs_input = (model['fc1.weight'].abs().sum(axis=1)) / l1_nonzero_input
    l1_nz_abs_output = (model['fc2.weight'].abs().sum(axis=0)) / l1_nonzero_output
    l2_nz_abs_input = (model['fc2.weight'].abs().sum(axis=1)) / l2_nonzero_input
    l2_nz_abs_output = (model['fc3.weight'].abs().sum(axis=0)) / l2_nonzero_output

    l1 = pd.DataFrame({
        "nz_cnt_input": l1_nonzero_input, "nz_cnt_output": l1_nonzero_output,
        "nz_pos_input": l1_positive_input, "nz_pos_output": l1_positive_output,
        "nz_avg_input": l1_nz_avg_input, "nz_avg_output": l1_nz_avg_output,
        "nz_abs_input": l1_nz_abs_input, "nz_abs_output": l1_nz_abs_output,
    })
    l2 = pd.DataFrame({
        "nz_cnt_input": l2_nonzero_input, "nz_cnt_output": l2_nonzero_output,
        "nz_pos_input": l2_positive_input, "nz_pos_output": l2_positive_output,
        "nz_avg_input": l2_nz_avg_input, "nz_avg_output": l2_nz_avg_output,
        "nz_abs_input": l2_nz_abs_input, "nz_abs_output": l2_nz_abs_output,
    })

    scatterplot(l1, 'nz_cnt_input', 'nz_cnt_output', "Layer 1: nonzero input vs output weights", "plots/scatter_l1_nnz.jpg")
    scatterplot(l2, 'nz_cnt_input', 'nz_cnt_output', "Layer 2: nonzero input vs output weights", "plots/scatter_l2_nnz.jpg")
    scatterplot(l1, 'nz_abs_input', 'nz_abs_output', "Layer 1: average absolute value of input vs output weights", "plots/scatter_l1_nz_abs.jpg")
    scatterplot(l2, 'nz_avg_input', 'nz_avg_output', "Layer 2: average value of input vs output weights", "plots/scatter_l2_nz_avg.jpg")
    scatterplot(l1, 'nz_avg_input', 'nz_avg_output', "Layer 1: average value of input vs output weights", "plots/scatter_l1_nz_avg.jpg")
    scatterplot(l2, 'nz_abs_input', 'nz_abs_output', "Layer 2: average absolute value of input vs output weights", "plots/scatter_l2_nz_abs.jpg")
    scatterplot(l1, 'nz_pos_input', 'nz_pos_output', "Layer 1: fraction of nonzero weights > 0 for input vs output", "plots/scatter_l1_pos.jpg")
    scatterplot(l2, 'nz_pos_input', 'nz_pos_output', "Layer 2: fraction of nonzero weights > 0 for input vs output", "plots/scatter_l2_pos.jpg")

    for col in l1.columns:
        l1[col].plot.hist(bins=20, title=f"L1 {col}").get_figure().savefig(f'plots/hist_l1_{col}.jpg')
        plt.clf()
        l2[col].plot.hist(bins=20, title=f"L2 {col}").get_figure().savefig(f'plots/hist_l2_{col}.jpg')
        plt.clf()

    print("layer 1 correlation:")
    print(l1.corr())
    print("layer 2 correlation:")
    print(l2.corr())

def scatterplot(df, xcol, ycol, title, outfile):
    print(df[[xcol, ycol]].corr())
    df_counts = df[[xcol, ycol]].groupby([xcol, ycol]).size().reset_index().rename(columns={0: 'count'})
    df_counts.plot.scatter(xcol, ycol, s=df_counts['count'], title=title).get_figure().savefig(outfile)
    plt.clf()
    
if __name__ == '__main__':
    analyze_feedforward_mnist()
