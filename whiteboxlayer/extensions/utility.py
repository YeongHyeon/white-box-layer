import tensorflow as tf

def trim_odd(layer, x):

    b, t, c = list(x.get_shape())
    if(int(t) % 2 == 1):
        return tf.slice(x, [0, 0, 0], [-1, int(t)-1, -1])
    else:
        return x

def trim_shape(layer, x, shape):

    return tf.slice(x, [0, 0, 0], shape)

def sub_pixel1d(layer, x, ratio, verbose=True):

    y1 = tf.transpose(a=x, perm=[2, 1, 0]) # (r, w, b)
    y2 = tf.batch_to_space(y1, [ratio], [[0, 0]]) # (1, r*w, b)
    y3 = tf.transpose(a=y2, perm=[2, 1, 0])

    if(verbose): print("SubPixel", x.shape, "->", y3.shape)
    return y3
