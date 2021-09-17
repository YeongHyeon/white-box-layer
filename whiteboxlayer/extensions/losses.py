import tensorflow as tf

def loss_l1(x, reduce=None):

    distance = tf.math.reduce_mean(\
        tf.math.abs(x), axis=reduce)

    return distance

def loss_l2(x, reduce=None):

    distance = tf.math.reduce_mean(\
        tf.math.sqrt(\
            tf.math.square(x) + 1e-30), axis=reduce)

    return distance

def loss_l2_log(x, reduce=None):

    distance = tf.math.reduce_mean(\
        -tf.math.log(\
            1-tf.math.sqrt(\
                tf.math.square(x) + 1e-30) + 1e-30), axis=reduce)

    return distance
