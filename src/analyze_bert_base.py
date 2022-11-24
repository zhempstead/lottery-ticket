import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def analyze_bert_layer(model, layer_num):
    pass

    
    
def basic_weight_statistics(param_array):
    
    # number of non-zero weights per input dimension
    nonzero_outgoing = (param_array != 0).sum(axis=0)
    # number of non-zero weights per output dimension
    nonzero_incoming = (param_array != 0).sum(axis=1)

    # number of positive weights per input dimension
    positive_outgoing = (param_array > 0).sum(axis=0)
    # number of positive weights per output dimension
    positive_incoming = (param_array > 0).sum(axis=1)

    # non-zero average weight per input dimension
    nz_avg_outgoing = (param_array.sum(axis=0)) / nonzero_outgoing
    # non-zero average weight per output dimension
    nz_avg_incoming = (param_array.sum(axis=1)) / nonzero_incoming

    # non-zero absolute average weight per input dimension
    nz_abs_avg_outgoing = (np.abs(param_array).sum(axis=0)) / nonzero_outgoing
    # non-zero absolute average weight per output dimension
    nz_abs_avg_incoming = (np.abs(param_array).sum(axis=1)) / nonzero_incoming

    outgoing_df = pd.DataFrame({
        'nonzero': nonzero_outgoing,
        'positive': positive_outgoing,
        'nz_avg': nz_avg_outgoing,
        'nz_abs_avg': nz_abs_avg_outgoing
    })
    incoming_df = pd.DataFrame({
        'nonzero': nonzero_incoming,
        'positive': positive_incoming,
        'nz_avg': nz_avg_incoming,
        'nz_abs_avg': nz_abs_avg_incoming
    })

    return outgoing_df, incoming_df
