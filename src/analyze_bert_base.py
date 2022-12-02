import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch






def analyze_bert_ffn(model, layer_num, PRUNE_AMT, PLOT_OPTION=False):
    layer = model.encoder.layer[layer_num]
    attn_out = layer.attention.output.dense.weight.data.numpy()
    ffn_inter = layer.intermediate.dense.weight.data.numpy()
    ffn_out = layer.output.dense.weight.data.numpy()
    
    hidden1_df, hidden2_df = compute_ffn_weight_stats(attn_out, ffn_inter, ffn_out)

    print("Hidden layer 1:")
    hidden1_corr = compute_input_output_corr(hidden1_df)
    print("Hidden layer 2:")
    hidden2_corr =  compute_input_output_corr(hidden2_df)

    if PLOT_OPTION == True:
        scatterplot(hidden1_df, 'nz_cnt_input', 'nz_cnt_output', f"Layer {layer_num}: Hidden 1: nonzero input vs output weights",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_scatter_hidden1_df_nnz.jpg")
        scatterplot(hidden2_df, 'nz_cnt_input', 'nz_cnt_output',
                    f"Layer {layer_num}: Hidden 2: nonzero input vs output weights", f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_df_nnz.jpg")
        scatterplot(hidden2_df, 'nz_avg_input', 'nz_avg_output',
                    f"Layer {layer_num}: Hidden 2: average value of input vs output weights", f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_df_nz_avg.jpg")
        scatterplot(hidden1_df, 'nz_avg_input', 'nz_avg_output',
                    f"Layer {layer_num}: Hidden 1: average value of input vs output weights", f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden1_df_nz_avg.jpg")
        scatterplot(hidden1_df, 'nz_abs_input', 'nz_abs_output',
                    f"Layer {layer_num}: Hidden 1: average absolute value of input vs output weights", f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden1_df_nz_abs.jpg")
        scatterplot(hidden2_df, 'nz_abs_input', 'nz_abs_output',
                    f"Layer {layer_num}: Hidden 2: average absolute value of input vs output weights", f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_df_nz_abs.jpg")
        scatterplot(hidden1_df, 'nz_pos_input', 'nz_pos_output',
                    f"Layer {layer_num}: Hidden 1: fraction of nonzero weights > 0 for input vs output", f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden1_pos.jpg")
        scatterplot(hidden2_df, 'nz_pos_input', 'nz_pos_output',
                    f"Layer {layer_num}: Hidden 2: fraction of nonzero weights > 0 for input vs output", f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_pos.jpg")

    return hidden1_corr, hidden2_corr


def compute_ffn_weight_stats(attn_params, inter_params, output_params):
    
    hidden1_nz_input = (attn_params != 0).sum(axis=1)
    hidden1_nz_output = (inter_params != 0).sum(axis=0)
    hidden2_nz_input = (inter_params != 0).sum(axis=1)
    hidden2_nz_output = (output_params != 0).sum(axis=0)

    hidden1_positive_input = (attn_params > 0).sum(axis=1) / hidden1_nz_input
    hidden1_positive_output = (inter_params > 0).sum(axis=0) / hidden1_nz_output
    hidden2_positive_input = (inter_params > 0).sum(axis=1) / hidden2_nz_input
    hidden2_positive_output = (output_params > 0).sum(axis=0) / hidden2_nz_output

    hidden1_nz_avg_input = attn_params.sum(axis=1) / hidden1_nz_input
    hidden1_nz_avg_output = inter_params.sum(axis=0) / hidden1_nz_output
    hidden2_nz_avg_input = inter_params.sum(axis=1) / hidden2_nz_input
    hidden2_nz_avg_output = output_params.sum(axis=0) / hidden2_nz_output

    hidden1_nz_abs_input = np.abs(attn_params).sum(axis=1) / hidden1_nz_input
    hidden1_nz_abs_output = np.abs(inter_params).sum(axis=0) / hidden1_nz_output
    hidden2_nz_abs_input = (np.abs(inter_params).sum(axis=1)) / hidden2_nz_input
    hidden2_nz_abs_output = (np.abs(output_params).sum(axis=0)) / hidden2_nz_output

    hidden1_df = pd.DataFrame({
        'nz_cnt_input': hidden1_nz_input, "nz_cnt_output": hidden1_nz_output,
        'nz_pos_input': hidden1_positive_input, "nz_pos_output": hidden1_positive_output,
        'nz_avg_input': hidden1_nz_avg_input, "nz_avg_output": hidden1_nz_avg_output,
        'nz_abs_input': hidden1_nz_abs_input, "nz_abs_output": hidden1_nz_abs_output,
        })
    hidden2_df = pd.DataFrame({
        "nz_cnt_input": hidden2_nz_input, "nz_cnt_output": hidden2_nz_output,
        "nz_pos_input": hidden2_positive_input, "nz_pos_output": hidden2_positive_output,
        "nz_avg_input": hidden2_nz_avg_input, "nz_avg_output": hidden2_nz_avg_output,
        "nz_abs_input": hidden2_nz_abs_input, "nz_abs_output": hidden2_nz_abs_output,
        })

    return hidden1_df, hidden2_df


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


def compute_input_output_corr(df):
    nz_cnt_corr = np.corrcoef(df['nz_cnt_input'],
                              df['nz_cnt_output'])[0, 1]
    nz_pos_corr = np.corrcoef(df['nz_pos_input'],
                              df['nz_pos_output'])[0, 1]
    nz_avg_corr = np.corrcoef(df['nz_avg_input'],
                              df['nz_avg_output'])[0, 1]
    nz_abs_corr = np.corrcoef(df['nz_abs_input'],
                              df['nz_abs_output'])[0, 1]


    c_str = f"""    nz_cnt_corr: {nz_cnt_corr:.3f} 
    nz_pos_corr: {nz_pos_corr:.3f}
    nz_avg_corr: {nz_avg_corr:.3f}
    nz_abs_corr: {nz_abs_corr:.3f}
    """
    print(c_str)

    corr_lst = [nz_cnt_corr, nz_pos_corr, nz_avg_corr, nz_abs_corr]

    return corr_lst


def scatterplot(df, xcol, ycol, title, outfile=False):
    #print(df[[xcol, ycol]].corr())
    df_counts = df[[xcol, ycol]].groupby([xcol, ycol]).size().reset_index().rename(columns={0: 'count'})
    if outfile:
        df_counts.plot.scatter(xcol, ycol, s=df_counts['count'], title=title).get_figure().savefig(outfile)
        plt.clf()
    else:
        df_counts.plot.scatter(xcol, ycol, s=df_counts['count'], title=title).get_figure()


def array_heatmap(arr):
    # Function that plots a heatmap of a numpy array
    plt.imshow(arr, cmap='hot', interpolation='nearest')