import numpy as np
import tensorflow as tf

def trim_odd(x):

    b, t, c = list(x.get_shape())
    if(int(t) % 2 == 1):
        return tf.slice(x, [0, 0, 0], [-1, int(t)-1, -1])
    else:
        return x

def trim_shape(x, shape):

    return tf.slice(x, [0, 0, 0], shape)

def sub_pixel1d(x, ratio, verbose=True):

    y1 = tf.transpose(a=x, perm=[2, 1, 0]) # (r, w, b)
    y2 = tf.batch_to_space(y1, [ratio], [[0, 0]]) # (1, r*w, b)
    y3 = tf.transpose(a=y2, perm=[2, 1, 0])

    if(verbose): print("SubPixel", x.shape, "->", y3.shape)
    return y3

def attention(x):

    list_shape = x.get_shape().as_list()

    xr = tf.reshape(x, [list_shape[0], -1])
    y = tf.reshape(tf.nn.softmax(xr, axis=1), list_shape)

    return y

def get_allweight(model):

    list_pkey = list(model.layer.parameters.keys())
    w_stack = []
    for idx_pkey, name_pkey in enumerate(list_pkey):
        w_stack.append(model.layer.parameters[name_pkey].numpy().flatten())

    return np.hstack(w_stack) # as vector

def set_allweight(model, new_weight):

    list_pkey = list(model.layer.parameters.keys())
    idx_numparam = 0
    for idx_pkey, name_pkey in enumerate(list_pkey):
        tmp_shape = list(model.layer.parameters[name_pkey].shape)

        tmp_numparam = 1
        for val_shape in tmp_shape:
            tmp_numparam *= val_shape

        tmp_constant = model.layer.parameters[name_pkey].numpy() * 0
        model.layer.parameters[name_pkey].assign(new_weight[idx_numparam:idx_numparam+tmp_numparam].reshape(tmp_shape))
        idx_numparam += tmp_numparam

    return model
