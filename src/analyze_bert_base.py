import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def analyze_bert_ffn(model, layer_num):
    layer = model.encoder.layer[layer_num]
    intermediate = layer.intermediate.dense.weight.data.numpy() # hidden
    output = layer.output.dense.weight.data.numpy()
    
    outgoing_df, incoming_df = \
        basic_weight_statistics(
            model.encoder.layer[0].attention.self.query.weight.detach().numpy())
    #query, key, value weights



def compute_ffn_weight_statistics:

def analyze_bert_self_attn(model, layer_num):
    layer = model.encoder.layer[layer_num]
    query_weights = layer.attention.self.query.weight.data.numpy()
    key_weights = layer.attention.self.key.weight.data.numpy()
    value_weights = layer.attention.self.value.weight.data.numpy()
    attn_output = layer. attention.output.dense.weight.data.numpy()
    
    
    q_out_df, q_in_df = compute_basic_weight_stats(query_weights)
    k_out_df, k_in_df = compute_basic_weight_stats(key_weights)
    v_out_df, v_in_df = compute_basic_weight_stats(value_weights)
    ffn_out_df, ffn_in_df = compute_basic_weight_stats(attn_output)

    q_dead_out_nodes = sum(q_out_df['nonzero'] == 0)
    q_dead_in_nodes = sum(q_in_df['nonzero'] == 0)
    #TODO fill out the rest of this functio for key, value, ffn
    
    
    
def compute_basic_weight_stats(param_array):
    """
    Function for computing basic weight statistics of a single parameter array

    Input
        param_array (numpy array) : two dimensional array of weights

    Return
        outgoing_df (pandas df) : two dimensional df of stats for weights
                                    corresponding to inputs to the param_array
        incoming_df (pandas df) : two dimensional df of stats for weights
                                    corresponding to outputs of the param_array
    """
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
