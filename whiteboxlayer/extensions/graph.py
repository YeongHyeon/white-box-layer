import numpy as np
import tensorflow as tf

def graph_conv(layer, x, a, c_out, \
    batch_norm=False, activation=None, name='', verbose=True):

    c_in, c_out = x.get_shape().as_list()[-1], int(c_out)

    w = layer.get_variable(shape=[c_in, c_out], \
        trainable=True, name='%s_w' %(name))
    b = layer.get_variable(shape=[c_out], \
        trainable=True, name='%s_b' %(name))

    wx = tf.linalg.matmul(x, w, name='%s_mul' %(name))
    y_feat = tf.math.add(wx, b, name='%s_add' %(name))
    y = tf.linalg.matmul(a, y_feat)

    if(verbose): print("G-Conv (%s)" %(name), x.shape, "->", y.shape)
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def graph_attention(layer, x, c_out, \
    batch_norm=False, activation=None, name='', verbose=True):

    wx = layer.fully_connected(x=x, c_out=c_out, \
        batch_norm=False, activation=None, name=name, verbose=False)

    y = tf.math.reduce_sum(wx, axis=-1)

    if(verbose): print("Readout (%s)" %(name), x.shape, "->", y.shape)
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def read_out(layer, x, c_out, \
    batch_norm=False, activation=None, name='', verbose=True):

    wx = layer.fully_connected(x=x, c_out=c_out, \
        batch_norm=False, activation=None, name=name, verbose=False)

    y = tf.math.reduce_sum(wx, axis=-1)

    if(verbose): print("Readout (%s)" %(name), x.shape, "->", y.shape)
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def pipgcn_node_average(layer, node, c_out, \
    batch_norm=False, activation=None, name='', verbose=True):

    node_in, c_out = node.get_shape().as_list()[-1], int(c_out)

    w_node_c = layer.get_variable(shape=[node_in, c_out], \
        trainable=True, name='%s_w_node_c' %(name))
    b = layer.get_variable(shape=[c_out], \
        trainable=True, name='%s_b' %(name))

    """ -=-=-= Term 1: node aggregation =-=-=- """
    term1 = tf.linalg.matmul(node, w_node_c, name='%s_term1' %(name)) # N x c_out

    y = term1 + b
    if(verbose): print("N-Avg (%s)" %(name), node.shape, "->", y.shape)
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def pipgcn_node_edge_average(layer, node, edge, hood, c_out, \
    batch_norm=False, activation=None, name='', verbose=True):

    node_in, c_out = node.get_shape().as_list()[-1], int(c_out)
    edge_in = edge.get_shape().as_list()[-1]
    hood = tf.squeeze(hood, axis=2)
    hood_in = tf.expand_dims(tf.math.count_nonzero(hood + 1, axis=1, dtype=tf.float32), -1)

    w_node_c = layer.get_variable(shape=[node_in, c_out], \
        trainable=True, name='%s_w_node_c' %(name))
    w_node_n = layer.get_variable(shape=[node_in, c_out], \
        trainable=True, name='%s_w_node_n' %(name))
    w_edge = layer.get_variable(shape=[edge_in, c_out], \
        trainable=True, name='%s_w_edge' %(name))
    b = layer.get_variable(shape=[c_out], \
        trainable=True, name='%s_b' %(name))

    """ -=-=-= Term 1: node aggregation =-=-=- """
    term1 = tf.linalg.matmul(node, w_node_c, name='%s_term1' %(name)) # N x c_out

    """ -=-=-= Term 2: edge aggregation =-=-=- """
    wn = tf.linalg.matmul(node, w_node_n, name='%s_term2_wn' %(name)) # N x c_out
    we = tf.linalg.matmul(edge, w_edge, name='%s_term2_we' %(name))  # N x num_edge x c_out
    gather_n = tf.gather(wn, hood)
    node_avg = tf.reduce_sum(gather_n, 1)
    edge_avg = tf.reduce_sum(we, 1)
    numerator = node_avg + edge_avg
    denominator = tf.maximum(hood_in, tf.ones_like(hood_in))
    term2 = tf.math.divide(numerator, denominator)  # (n_verts, v_filters)

    y = term1 + term2 + b
    if(verbose): print("N-E-Avg (%s)" %(name), node.shape, "->", y.shape)
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def pipgcn_order_dependent(layer, node, edge, hood, c_out, \
    batch_norm=False, activation=None, name='', verbose=True):

    node_in, c_out = node.get_shape().as_list()[-1], int(c_out)
    edge_in_o = edge.get_shape().as_list()[-2]
    edge_in_f = edge.get_shape().as_list()[-1]
    hood = tf.squeeze(hood, axis=2)
    hood_in = tf.expand_dims(tf.math.count_nonzero(hood + 1, axis=1, dtype=tf.float32), -1)

    w_node_c = layer.get_variable(shape=[node_in, c_out], \
        trainable=True, name='%s_w_node_c' %(name))
    w_node_n = layer.get_variable(shape=[node_in, c_out], \
        trainable=True, name='%s_w_node_n' %(name))
    w_edge_order = layer.get_variable(shape=[edge_in_o, c_out], \
        trainable=True, name='%s_w_edge_o' %(name))
    w_edge_feat = layer.get_variable(shape=[edge_in_f, c_out], \
        trainable=True, name='%s_w_edge_f' %(name))
    b = layer.get_variable(shape=[c_out], \
        trainable=True, name='%s_b' %(name))

    """ -=-=-= Term 1: node aggregation =-=-=- """
    term1 = tf.linalg.matmul(node, w_node_c, name='%s_term1' %(name)) # N x c_out

    """ -=-=-= Term 2: edge aggregation =-=-=- """
    wn = tf.linalg.matmul(node, w_node_n, name='%s_term2_wn' %(name)) # N x c_out
    we_o = tf.linalg.matmul(tf.transpose(edge, perm=[0, 2, 1]), w_edge_order, name='%s_term2_we_o' %(name))  # N x num_edge x c_out
    we_f = tf.linalg.matmul(edge, w_edge_feat, name='%s_term2_we_f' %(name))  # N x num_edge x c_out
    gather_n = tf.gather(wn, hood)
    node_avg = tf.reduce_sum(gather_n, 1)
    edge_order = tf.reduce_sum(we_o, 1) + tf.reduce_sum(we_f, 1)
    numerator = node_avg + edge_order
    denominator = tf.maximum(hood_in, tf.ones_like(hood_in))
    term2 = tf.math.divide(numerator, denominator)  # (n_verts, v_filters)

    y = term1 + term2 + b
    if(verbose): print("Order-N-E-Avg (%s)" %(name), node.shape, "->", y.shape)
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def merge_ligand_receptor(layer, ligand, receptor, pair, verbose=True):

    side_lgnd = tf.gather(ligand, pair[:, 0]) # select ligand node via pair minibatch
    side_rcpt = tf.gather(receptor, pair[:, 1]) # select receptor node via pair minibatch

    y = tf.concat([side_lgnd, side_rcpt], axis=1)
    if(verbose): print("Merge", ligand.shape, receptor.shape, "->", y.shape)
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return y
