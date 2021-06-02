import numpy as np
import tensorflow as tf

def lstm_cell(layer, x_now, h_prev, c_prev, output_len, \
    activation="tanh", recurrent_activation="sigmoid", name='', verbose=True):

    if(h_prev is None): h_prev = tf.zeros_like(x_now)
    if(c_prev is None): c_prev = tf.zeros_like(x_now)

    i_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-i-term1" %(name), verbose=verbose)

    i_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-i-term1" %(name), verbose=verbose)
    i_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-i-term2" %(name), verbose=verbose)
    i_now = layer.fully_connected(x=i_term1 + i_term2, c_out=x_now.shape[-1], \
        batch_norm=False, activation=recurrent_activation, name="%s-i" %(name), verbose=verbose)

    f_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-f-term1" %(name), verbose=verbose)
    f_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-f-term2" %(name), verbose=verbose)
    f_now = layer.fully_connected(x=f_term1 + f_term2, c_out=x_now.shape[-1], \
        batch_norm=False, activation=recurrent_activation, name="%s-f" %(name), verbose=verbose)

    o_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-o-term1" %(name), verbose=verbose)
    o_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-o-term2" %(name), verbose=verbose)
    o_now = layer.fully_connected(x=o_term1 + o_term2, c_out=x_now.shape[-1], \
        batch_norm=False, activation=recurrent_activation, name="%s-o" %(name), verbose=verbose)

    c_term1_1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-c-term1_1" %(name), verbose=verbose)
    c_term1_2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-c-term2_1" %(name), verbose=verbose)
    c_term1_sum = layer.activation(x=c_term1_1 + c_term1_2, \
        activation="tanh", name="%s-c-term1-sum" %(name))

    c_term1 = tf.compat.v1.multiply(i_now, c_term1_sum, name="%s-c-term1" %(name))
    c_term2 = tf.compat.v1.multiply(f_now, c_prev, name="%s-c-term2" %(name))
    c_now = tf.compat.v1.add(c_term1, c_term2, name="%s-c" %(name))

    h_now = tf.compat.v1.multiply(o_now, \
        layer.activation(x=c_now, activation=activation, name="%s-c-act" %(name)), \
        name="%s-h" %(name))

    y_now = layer.fully_connected(x=h_now, c_out=output_len, \
        batch_norm=False, activation=activation, name="%s-y" %(name), verbose=verbose)

    if(verbose): print("LSTM Cell (%s)" %(name), x_now.shape, "->", y_now.shape)
    return h_now, c_now, y_now

def lstm_layer(layer, x, output_len, \
    batch_norm=False, activation="tanh", recurrent_activation="sigmoid", name='', verbose=True):

    x = tf.transpose(x, perm=[1, 0, 2])
    dim_seq = x.get_shape().as_list()[0]
    y, h_now, c_now = None, None, None
    for idx_s in range(dim_seq):

        h_now, c_now, x_new = lstm_cell(layer=layer, \
            x_now=x[idx_s, :, :], h_prev=h_now, c_prev=c_now, \
            output_len=output_len, activation=activation, recurrent_activation=recurrent_activation, \
            name=name, verbose=(verbose and idx_s == 0))

        x_new = tf.expand_dims(x_new, 0)
        if(y is None): y = x_new
        else: y = tf.concat([y, x_new], 0)

    y = tf.transpose(y, perm=[1, 0, 2])
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return y
