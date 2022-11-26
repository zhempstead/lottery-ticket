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



def compute_ffn_weight_statistics(inter_params, output_params):
    
    hidden_nz_input = (inter_params != 0).sum(axis=1)
    hidden_nz_output = (output_params != 0).sum(axis=0)

    hidden_positive_input = (inter_params > 0).sum(axis=1) / hidden_nz_input
    hidden_positive_output = (output_params > 0).sum(axis=0) / hidden_nz_output

    hidden_nz_avg_input = (inter_params.sum(axis=1)) / hidden_nz_input
    hidden_nz_avg_output = (output_params.sum(axis=0)) / hidden_nz_output

    hidden_nz_abs_input = (np.abs(inter_params).sum(axis=1)) / hidden_nz_input
    hidden_nz_abs_output = (np.abs(output_params).sum(axis=0)) / hidden_nz_output

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
    k_dead_out_nodes = sum(k_out_df['nonzero'] == 0)
    k_dead_in_nodes = sum(k_in_df['nonzero'] == 0)
    v_dead_out_nodes = sum(v_out_df['nonzero'] == 0)
    v_dead_in_nodes = sum(v_in_df['nonzero'] == 0)
    ffn_dead_out_nodes = sum(ffn_out_df['nonzero'] == 0)
    ffn_dead_in_nodes = sum(ffn_in_df['nonzero'] == 0)

    dead_node_df = pd.DataFrame({
        "q_weights": [q_dead_out_nodes, q_dead_in_nodes],
        "k_weights": [k_dead_out_nodes, k_dead_in_nodes],
        "v_weights": [v_dead_out_nodes, v_dead_in_nodes],
        "ffn_weights": [ffn_dead_out_nodes, ffn_dead_in_nodes]
    }, index=["output", "input"])

    return dead_node_df

    
    
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
