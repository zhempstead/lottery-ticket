import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import torch


NUM_LAYERS = 12
NUM_HEADS = 12


def analyze_bert_layerwise(model, PRUNE_AMT, PLOT_OPTION=False):
    """
    Function for analyzing the BERT model
    Inputs
        model: BERT model
        PRUNE_AMT: prune ratio
        PLOT_OPTION: boolean for plotting
    Outputs
        hidden1_corr_lst: list of correlations for hidden layer 1
        hidden2_corr_lst: list of correlations for hidden layer 2
    """
    # Layer by layer analysis
    ffn_corr1_df = df = pd.DataFrame({'nz_cnt_corr:': pd.Series(dtype='float'),
                   'nz_pos_corr': pd.Series(dtype='float'),
                   'nz_avg_corr': pd.Series(dtype='float'),
                   'nz_abs_corr:': pd.Series(dtype='float')})

    ffn_corr2_df = df = pd.DataFrame({'nz_cnt_corr:': pd.Series(dtype='float'),
                   'nz_pos_corr': pd.Series(dtype='float'),
                   'nz_avg_corr': pd.Series(dtype='float'),
                   'nz_abs_corr:': pd.Series(dtype='float')})


    raw_corr_df = pd.DataFrame(columns= ['layer_number','raw_q_k_corr',
                                            'raw_q_v_corr', 'raw_k_v_corr'])
    head_raw_corr_df = pd.DataFrame(columns= ['layer_number','raw_q_k_corr',
                                                'raw_q_v_corr', 'raw_k_v_corr',
                                                'head_number'])

    summ_attn_corr_colnames = ['layer_number','q_k_nonzero', 'q_v_nonzero',
                                'k_v_nonzero','q_k_positive', 'q_v_positive',
                                'k_v_positive','q_k_nz_avg', 'q_v_nz_avg',
                                'k_v_nz_avg', 'q_k_nz_abs_avg','q_v_nz_abs_avg',
                                'k_v_nz_abs_avg']
    summ_attn_corr_in_df = pd.DataFrame(columns=summ_attn_corr_colnames)
    summ_attn_corr_out_df = pd.DataFrame(columns=summ_attn_corr_colnames)
    head_summ_attn_corr_in_df = pd.DataFrame(columns=summ_attn_corr_colnames + ['head_number']) # for headwise analysis
    head_summ_attn_corr_out_df = pd.DataFrame(columns=summ_attn_corr_colnames + ['head_number'])

    for layer_num in range(12):
        print(f"Feed-Forward Node Input/Output Correlations for layer {layer_num}:")
        rv_corr1, rv_corr2 = analyze_bert_ffn_layer(model, layer_num,
                                                    PRUNE_AMT=0,
                                                    PLOT_OPTION=False)
        ffn_corr1_df.loc[len(ffn_corr1_df)] = rv_corr1
        ffn_corr2_df.loc[len(ffn_corr2_df)] = rv_corr2
        
        layer_analysis_dict = analyze_bert_self_attn_layer(model,layer_num)
        raw_corr_df = pd.concat([raw_corr_df,layer_analysis_dict['raw_corr']],
                                ignore_index=True)
        head_raw_corr_df = pd.concat([head_raw_corr_df,
                                      layer_analysis_dict['head_raw_corr']],
                                     ignore_index=True)
        summ_attn_corr_in_df = pd.concat([summ_attn_corr_in_df,
                                          pd.DataFrame(layer_analysis_dict['summ_attn_corr_in'],
                                                        index=[layer_num])],
                                         ignore_index=True) 
        summ_attn_corr_out_df = pd.concat([summ_attn_corr_out_df,
                                           pd.DataFrame(layer_analysis_dict['summ_attn_corr_out'],
                                                        index=[layer_num])],
                                         ignore_index=True)
        head_summ_attn_corr_in_df = pd.concat([head_summ_attn_corr_in_df,
                                               layer_analysis_dict['head_summ_attn_corr_in']],
                                                ignore_index=True)
        head_summ_attn_corr_out_df = pd.concat([head_summ_attn_corr_out_df,
                                               layer_analysis_dict['head_summ_attn_corr_out']],
                                               ignore_index=True)

    return ffn_corr1_df, ffn_corr2_df, raw_corr_df, head_raw_corr_df, \
            summ_attn_corr_in_df, summ_attn_corr_out_df, \
                head_summ_attn_corr_in_df, head_summ_attn_corr_out_df


def analyze_bert_ffn_layer(model, layer_num, PRUNE_AMT, PLOT_OPTION=False):
    """
    Function for analyzing the feed forward portion of a single BERT layer
    Inputs
        model: BERT model
        layer_num: layer number to analyze
        PRUNE_AMT: prune ratio
        PLOT_OPTION: boolean for plotting
    Outputs
        hidden1_corr_lst: list of correlations for hidden layer 1
        hidden2_corr_lst: list of correlations for hidden layer 2
    """
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
        scatterplot(hidden1_df, 'nz_cnt_input', 'nz_cnt_output',
                    f"Layer {layer_num}: FFN Hidden : nonzero input vs output weights",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_scatter_hidden1_df_nnz.jpg")
        scatterplot(hidden2_df, 'nz_cnt_input', 'nz_cnt_output',
                    f"Layer {layer_num}: FFN Hidden 2: nonzero input vs output weights",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_df_nnz.jpg")
        scatterplot(hidden2_df, 'nz_avg_input', 'nz_avg_output',
                    f"Layer {layer_num}: FFN Hidden 2: average value of input vs output weights",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_df_nz_avg.jpg")
        scatterplot(hidden1_df, 'nz_avg_input', 'nz_avg_output',
                    f"Layer {layer_num}: FFN Hidden `: average value of input vs output weights",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden1_df_nz_avg.jpg")
        scatterplot(hidden1_df, 'nz_abs_input', 'nz_abs_output',
                    f"Layer {layer_num}: FFN Hidden `: average absolute value of input vs output weights",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden1_df_nz_abs.jpg")
        scatterplot(hidden2_df, 'nz_abs_input', 'nz_abs_output',
                    f"Layer {layer_num}: FFN FFN Hidden 2: average absolute value of input vs output weights",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_df_nz_abs.jpg")
        scatterplot(hidden1_df, 'nz_pos_input', 'nz_pos_output',
                    f"Layer {layer_num}: FFN Hidden : fraction of nonzero weights > 0 for input vs output",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden1_pos.jpg")
        scatterplot(hidden2_df, 'nz_pos_input', 'nz_pos_output',
                    f"Layer {layer_num}: FFN Hidden 2: fraction of nonzero weights > 0 for input vs output",
                    f"plots/BERT_{PRUNE_AMT}_{layer_num}_ffn_scatter_hidden2_pos.jpg")

    return hidden1_corr, hidden2_corr


def compute_ffn_weight_stats(attn_params, inter_params, output_params):
    """
    Function for computing statistics of input and output weights
    of the feed forward portion of a single BERT layer
    Inputs
        attn_params: numpy array of attention output weights
        inter_params: numpy array of intermediate weights
        output_params: numpy array of output weights
    Outputs
        hidden1_df: pandas dataframe of statistics for hidden layer 1
        hidden2_df: pandas dataframe of statistics for hidden layer 2
    """
    
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


def analyze_bert_self_attn_layer(model, layer_num):
    layer = model.encoder.layer[layer_num]
    query_weights = layer.attention.self.query.weight.data.numpy()
    key_weights = layer.attention.self.key.weight.data.numpy()
    value_weights = layer.attention.self.value.weight.data.numpy()

    out_dim = query_weights.shape[0]
    in_dim = query_weights.shape[1]
    head_dim = in_dim // NUM_HEADS

    head_raw_corr_df = pd.DataFrame(columns= ['layer_number',
                                              'raw_q_k_corr',
                                              'raw_q_v_corr',
                                              'raw_k_v_corr',
                                              'head_number'])

    head_summ_attn_corr_colnames = ['layer_number','head_number', 'q_k_nonzero',
                                    'q_v_nonzero','k_v_nonzero','q_k_positive',
                                    'q_v_positive','k_v_positive','q_k_nz_avg',
                                    'q_v_nz_avg','k_v_nz_avg', 'q_k_nz_abs_avg',
                                    'q_v_nz_abs_avg','k_v_nz_abs_avg']
    head_summ_attn_corr_in_df = pd.DataFrame(columns=head_summ_attn_corr_colnames)
    head_summ_attn_corr_out_df = pd.DataFrame(columns=head_summ_attn_corr_colnames)

    # head-level analysis
    for i in range(NUM_HEADS):
        query_head = query_weights[i*head_dim:(i+1)*head_dim, :]
        key_head = key_weights[i*head_dim:(i+1)*head_dim, :]
        value_head = value_weights[i*head_dim:(i+1)*head_dim, :]

        head_raw_corr, head_summary_corr_in, \
            head_summary_corr_out = analyze_key_query_value_weights(query_head,
                                                                    key_head,
                                                                    value_head,
                                                                    layer_num,
                                                                    head_num=i+1) # 1-indexed for plotting
        head_raw_corr_df = pd.concat([head_raw_corr_df, head_raw_corr],
                                     ignore_index=True)
        # print("ADDING HEAD SUMMARY CORR IN:")
        # print(head_summ_attn_corr_in_df)
        head_summ_attn_corr_in_df = pd.concat([head_summ_attn_corr_in_df,
                                          pd.DataFrame(head_summary_corr_in, index=[layer_num])],
                                        ignore_index=True) 
        head_summ_attn_corr_out_df = pd.concat([head_summ_attn_corr_out_df,
                                           pd.DataFrame(head_summary_corr_out, index=[layer_num])],
                                        ignore_index=True)
    
    # layer-level analysis
    layer_raw_corr, layer_summary_corr_in, \
        layer_summary_corr_out = analyze_key_query_value_weights(query_weights,
                                                                    key_weights,
                                                                    value_weights,
                                                                    layer_num)

    # q_dead_out_nodes = sum(q_out_df['nonzero'] == 0)
    # q_dead_in_nodes = sum(q_in_df['nonzero'] == 0)
    # k_dead_out_nodes = sum(k_out_df['nonzero'] == 0)
    # k_dead_in_nodes = sum(k_in_df['nonzero'] == 0)
    # v_dead_out_nodes = sum(v_out_df['nonzero'] == 0)
    # v_dead_in_nodes = sum(v_in_df['nonzero'] == 0)
    # ffn_dead_out_nodes = sum(ffn_out_df['nonzero'] == 0)
    # ffn_dead_in_nodes = sum(ffn_in_df['nonzero'] == 0)

    # dead_node_df = pd.DataFrame({
    #     "q_weights": [q_dead_out_nodes, q_dead_in_nodes],
    #     "k_weights": [k_dead_out_nodes, k_dead_in_nodes],
    #     "v_weights": [v_dead_out_nodes, v_dead_in_nodes],
    #     "ffn_weights": [ffn_dead_out_nodes, ffn_dead_in_nodes]
    # }, index=["output", "input"])

    layer_dict = {
        "layer_number": layer_num,"raw_corr": layer_raw_corr,
        "summ_attn_corr_in": layer_summary_corr_in,
        "summ_attn_corr_out": layer_summary_corr_out,
        "head_raw_corr": head_raw_corr_df,
        "head_summ_attn_corr_in": head_summ_attn_corr_in_df,
        "head_summ_attn_corr_out": head_summ_attn_corr_out_df
    }

    return layer_dict


def analyze_key_query_value_weights(q_weights, k_weights, v_weights,
                                    layer_num, head_num=None):
    q_out_df, q_in_df = compute_basic_weight_stats(q_weights)
    k_out_df, k_in_df = compute_basic_weight_stats(k_weights)
    v_out_df, v_in_df = compute_basic_weight_stats(v_weights)

    raw_q_k_corr = np.corrcoef(q_weights.flatten(), k_weights.flatten())[0, 1]
    raw_q_v_corr = np.corrcoef(q_weights.flatten(), v_weights.flatten())[0, 1]
    raw_k_v_corr = np.corrcoef(k_weights.flatten(), v_weights.flatten())[0, 1]

    raw_corr = pd.DataFrame({
        "layer_number": [layer_num + 1], # layer number starts from 1
        "raw_q_k_corr": [raw_q_k_corr], "raw_q_v_corr": [raw_q_v_corr],
        "raw_k_v_corr": [raw_k_v_corr]
        })

    summary_corr_in, summary_corr_out = compute_self_attn_corr(q_out_df, q_in_df,
                                                  k_out_df, k_in_df,
                                                  v_out_df, v_in_df)
    summary_corr_in["layer_number"] = layer_num + 1
    summary_corr_out["layer_number"] = layer_num + 1

    if head_num: # if head_num is not None add head_num col to the dataframes
        raw_corr["head_number"] = head_num
        summary_corr_in["head_number"] = head_num
        summary_corr_out["head_number"] = head_num

    return raw_corr, summary_corr_in, summary_corr_out


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
    """
    Function for computing correlation between input and output weights
    of a single parameter array
    input df is the output of compute_ffn_weight_stats
    """
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


def compute_self_attn_corr(in_q_w, out_q_w, in_k_w, out_k_w,
                           in_v_w, out_v_w):

    in_corr_dict = {}
    out_corr_dict = {}
    measures = ['nonzero', 'positive', 'nz_avg', 'nz_abs_avg']
    for measure in measures:
        q_k_label = f"q_k_{measure}"
        q_v_label = f"q_v_{measure}"
        k_v_label = f"k_v_{measure}"

        in_corr_dict[q_k_label] = np.corrcoef(in_q_w[measure], in_k_w[measure])[0, 1]
        in_corr_dict[q_v_label] = np.corrcoef(in_q_w[measure], in_v_w[measure])[0, 1]
        in_corr_dict[k_v_label] = np.corrcoef(in_k_w[measure], in_v_w[measure])[0, 1]

        out_corr_dict[q_k_label] = np.corrcoef(out_q_w[measure], out_k_w[measure])[0, 1]
        out_corr_dict[q_v_label] = np.corrcoef(out_q_w[measure], out_v_w[measure])[0, 1]
        out_corr_dict[k_v_label] = np.corrcoef(out_k_w[measure], out_v_w[measure])[0, 1]

    return in_corr_dict, out_corr_dict


def scatterplot(df, xcol, ycol, title, outfile=False):
    """
    create scatterplot
    Inputs:
      df: dataframe with data
      xcol: column name for x-axis
      ycol: column name for y-axis
      title: title for plot
      outfile: if True, save plot to file
    """
    df_counts = df[[xcol, ycol]].groupby([xcol, ycol]).size().reset_index().rename(columns={0: 'count'})
    if outfile:
        df_counts.plot.scatter(xcol, ycol, s=df_counts['count'], title=title).get_figure().savefig(outfile)
        #plt.clf()
    else:
        df_counts.plot.scatter(xcol, ycol, s=df_counts['count'], title=title).get_figure()


def array_heatmap(arr):
    # Function that plots a heatmap of a numpy array
    plt.imshow(arr, cmap='hot', interpolation='nearest')
