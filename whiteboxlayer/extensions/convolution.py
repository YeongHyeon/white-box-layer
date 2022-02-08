import tensorflow as tf

def conv1d(layer, x, stride, \
    filter_size=[3, 16, 32], dilations=[1, 1, 1], \
    padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

    w = layer.get_variable(shape=filter_size, \
        trainable=trainable, name='%s_w' %(name))
    if(usebias): b = layer.get_variable(shape=[filter_size[-1]], \
        trainable=trainable, name='%s_b' %(name))

    wx = tf.nn.conv1d(
        input=x,
        filters=w,
        stride=stride,
        padding=padding,
        data_format='NWC',
        dilations=None,
        name='%s_cv' %(name)
    )

    if(usebias): y = tf.math.add(wx, b, name='%s_add' %(name))
    else: y = wx
    if(verbose): print("Conv (%s)" %(name), x.shape, "->", y.shape)

    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=trainable, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def convt1d(layer, x, stride, output_shape, \
    filter_size=[3, 16, 32], dilations=[1, 1, 1], \
    padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

    w = layer.get_variable(shape=filter_size, \
        trainable=trainable, name='%s_w' %(name))
    if(usebias): b = layer.get_variable(shape=[filter_size[-2]], \
        trainable=trainable, name='%s_b' %(name))

    wx = tf.nn.conv1d_transpose(
        input=x,
        filters=w,
        output_shape=output_shape,
        strides=stride,
        padding=padding,
        data_format='NWC',
        dilations=dilations,
        name='%s_cvt' %(name)
    )

    if(usebias): y = tf.math.add(wx, b, name='%s_add' %(name))
    else: y = wx
    if(verbose): print("ConvT (%s)" %(name), x.shape, "->", y.shape)

    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=trainable, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def conv2d(layer, x, stride, \
    filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
    padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

    w = layer.get_variable(shape=filter_size, \
        trainable=trainable, name='%s_w' %(name))
    if(usebias): b = layer.get_variable(shape=[filter_size[-1]], \
        trainable=trainable, name='%s_b' %(name))

    wx = tf.nn.conv2d(
        input=x,
        filters=w,
        strides=[1, stride, stride, 1],
        padding=padding,
        data_format='NHWC',
        dilations=dilations,
        name='%s_cv' %(name)
    )

    if(usebias): y = tf.math.add(wx, b, name='%s_add' %(name))
    else: y = wx
    if(verbose): print("Conv (%s)" %(name), x.shape, "->", y.shape)

    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=trainable, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)

def convt2d(layer, x, stride, output_shape, \
    filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
    padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

    w = layer.get_variable(shape=filter_size, \
        trainable=trainable, name='%s_w' %(name))
    if(usebias): b = layer.get_variable(shape=[filter_size[-2]], \
        trainable=trainable, name='%s_b' %(name))

    wx = tf.nn.conv2d_transpose(
        input=x,
        filters=w,
        output_shape=output_shape,
        strides=[1, stride, stride, 1],
        padding=padding,
        data_format='NHWC',
        dilations=dilations,
        name='%s_cvt' %(name)
    )

    if(usebias): y = tf.math.add(wx, b, name='%s_add' %(name))
    else: y = wx
    if(verbose): print("ConvT (%s)" %(name), x.shape, "->", y.shape)

    if(batch_norm): y = layer.batch_normalization(x=y, \
        trainable=trainable, name='%s_bn' %(name), verbose=verbose)
    return layer.activation(x=y, activation=activation, name=name)
