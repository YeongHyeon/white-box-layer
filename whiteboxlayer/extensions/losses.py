import tensorflow as tf

def loss_ae(x, reduce=None):

    loss = tf.math.reduce_mean(\
        tf.math.abs(x), axis=reduce)

    return loss

def loss_mse(x, reduce=None):

    loss = tf.math.reduce_mean(\
        tf.math.square(x), axis=reduce)

    return loss

def loss_rmse(x, reduce=None):

    loss = tf.math.reduce_mean(\
        tf.math.sqrt(\
            tf.math.square(x) + 1e-30), axis=reduce)

    return loss

def loss_log_mse(x, reduce=None):

    loss = tf.math.reduce_mean(\
        -tf.math.log(\
            1 - tf.math.square(x) + 1e-30), axis=reduce)

    return loss

def loss_bce(true, pred, reduce=None):

    term1 = true * tf.math.log(pred + 1e-30)
    term2 = (1 - true) * tf.math.log(1 - pred + 1e-30)
    loss = tf.math.reduce_mean(-(term1 + term2), axis=reduce)

    return loss
