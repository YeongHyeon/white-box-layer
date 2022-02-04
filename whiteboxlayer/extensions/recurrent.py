import tensorflow as tf

def lstm_cell(layer, x_now, h_prev, c_prev, output_dim, \
    activation="tanh", recurrent_activation="sigmoid", name='', verbose=True):

    if(h_prev is None): h_prev = tf.zeros_like(x_now)
    if(c_prev is None): c_prev = tf.zeros_like(x_now)

    # input gate
    i_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-i-term1" %(name), verbose=verbose)
    i_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-i-term2" %(name), verbose=verbose)
    i_now = layer.activation(x=i_term1 + i_term2, \
        activation=recurrent_activation, name="%s-i" %(name))

    # forget gate
    f_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-f-term1" %(name), verbose=verbose)
    f_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-f-term2" %(name), verbose=verbose)
    f_now = layer.activation(x=f_term1 + f_term2, \
        activation=recurrent_activation, name="%s-f" %(name))

    # output gate
    o_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-o-term1" %(name), verbose=verbose)
    o_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-o-term2" %(name), verbose=verbose)
    o_now = layer.activation(x=o_term1 + o_term2, \
        activation=recurrent_activation, name="%s-o" %(name))

    # memory cell
    c_term1_1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-c-term1_1" %(name), verbose=verbose)
    c_term1_2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-c-term1_2" %(name), verbose=verbose)
    c_now_hat = layer.activation(x=c_term1_1 + c_term1_2, \
        activation=activation, name="%s-c-hat" %(name))

    c_term1 = tf.compat.v1.multiply(f_now, c_prev, name="%s-c-prev_mul" %(name))
    c_term2 = tf.compat.v1.multiply(i_now, c_now_hat, name="%s-c-hat_mul" %(name))
    c_now = tf.compat.v1.add(c_term1, c_term2, name="%s-c" %(name))

    c_now_act = layer.activation(x=c_now, activation=activation, name="%s-c-act" %(name))
    h_now = tf.compat.v1.multiply(o_now, c_now_act, name="%s-h" %(name))

    y_now = layer.fully_connected(x=h_now, c_out=output_dim, \
        batch_norm=False, activation=activation, name="%s-y" %(name), verbose=verbose)

    if(verbose): print("LSTM Cell (%s)" %(name), x_now.shape, "->", y_now.shape)
    return h_now, c_now, y_now

def lstm_layer(layer, x, output_dim, \
    batch_norm=False, activation="tanh", recurrent_activation="sigmoid", name='lstm', verbose=True):

    x = tf.transpose(x, perm=[1, 0, 2])
    dim_seq = x.get_shape().as_list()[0]
    y, h_now, c_now = None, None, None
    for idx_s in range(dim_seq):

        h_now, c_now, x_new = lstm_cell(layer=layer, \
            x_now=x[idx_s, :, :], h_prev=h_now, c_prev=c_now, output_dim=output_dim, \
            activation=activation, recurrent_activation=recurrent_activation, \
            name=name, verbose=(verbose and idx_s == 0))

        x_new = tf.expand_dims(x_new, 0)
        if(y is None): y = x_new
        else: y = tf.concat([y, x_new], 0)

    y = tf.transpose(y, perm=[1, 0, 2])
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return y

def gru_cell(layer, x_now, h_prev, output_dim, \
    activation="tanh", recurrent_activation="sigmoid", name='', verbose=True):

    if(h_prev is None): h_prev = tf.zeros_like(x_now)

    # update gate
    z_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-z-term1" %(name), verbose=verbose)
    z_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-z-term2" %(name), verbose=verbose)
    z_now = layer.activation(x=z_term1 + z_term2, \
        activation=recurrent_activation, name="%s-z" %(name))

    # reset gate
    r_term1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-r-term1" %(name), verbose=verbose)
    r_term2 = layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-r-term2" %(name), verbose=verbose)
    r_now = layer.activation(x=r_term1 + r_term2, \
        activation=recurrent_activation, name="%s-r" %(name))

    # candidate activation
    rh_prev = tf.compat.v1.multiply(r_now, h_prev, name="%s-rh-prev" %(name))
    h_term1_1 = layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
        batch_norm=False, activation=None, name="%s-h-term1_1" %(name), verbose=verbose)
    h_term1_2 = layer.fully_connected(x=rh_prev, c_out=h_prev.shape[-1], \
        batch_norm=False, activation=None, name="%s-h-term1_2" %(name), verbose=verbose)
    h_now_hat = layer.activation(x=h_term1_1 + h_term1_2, \
        activation=activation, name="%s-h-hat" %(name))

    h_term1 = tf.compat.v1.multiply(1-z_now, h_prev, name="%s-h_term1" %(name))
    h_term2 = tf.compat.v1.multiply(z_now, h_now_hat, name="%s-h_term2" %(name))
    h_now = tf.compat.v1.add(h_term1, h_term2, name="%s-c" %(name))

    y_now = layer.fully_connected(x=h_now, c_out=output_dim, \
        batch_norm=False, activation=activation, name="%s-y" %(name), verbose=verbose)

    if(verbose): print("LSTM Cell (%s)" %(name), x_now.shape, "->", y_now.shape)
    return h_now, y_now

def gru_layer(layer, x, output_dim, \
    batch_norm=False, activation="tanh", recurrent_activation="sigmoid", name='gru', verbose=True):

    x = tf.transpose(x, perm=[1, 0, 2])
    dim_seq = x.get_shape().as_list()[0]
    y, h_now = None, None
    for idx_s in range(dim_seq):

        h_now, x_new = gru_cell(layer=layer, \
            x_now=x[idx_s, :, :], h_prev=h_now, output_dim=output_dim, \
            activation=activation, recurrent_activation=recurrent_activation, \
            name=name, verbose=(verbose and idx_s == 0))

        x_new = tf.expand_dims(x_new, 0)
        if(y is None): y = x_new
        else: y = tf.concat([y, x_new], 0)

    y = tf.transpose(y, perm=[1, 0, 2])
    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=True, name='%s_bn' %(name), verbose=verbose)
    return y
