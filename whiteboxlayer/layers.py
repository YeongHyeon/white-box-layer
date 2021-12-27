import tensorflow as tf

class Layers(object):

    def __init__(self, parameters={}):

        self.num_params = 0
        self.parameters = {}
        self.initializer_xavier = tf.initializers.glorot_normal()

    """ -*-*-*-*-*- Variables -*-*-*-*-*- """

    def get_variable(self, shape=None, constant=None, trainable=True, name=''):

        try: return self.parameters[name]
        except:
            if(constant is None):
                w = tf.Variable(
                    initial_value=self.initializer_xavier(shape),
                    shape=shape,
                    trainable=trainable,
                    dtype=tf.float32,
                    name="%s" %(name)
                )
            else:
                w = tf.Variable(
                    initial_value=constant,
                    name="%s" %(name),
                    trainable=True,
                    dtype=tf.float32)

            tmp_num = 1
            for num in shape: tmp_num *= num
            self.num_params += tmp_num
            self.parameters[name] = w

            return self.parameters[name]

    """ -*-*-*-*-*- Classic Functions -*-*-*-*-*- """
    def activation(self, x, activation=None, name=''):

        name = "%s_act" %(name)
        if(activation is None): return x
        elif("sigmoid" == activation):
            return tf.nn.sigmoid(x, name='%s' %(name))
        elif("tanh" == activation):
            return tf.nn.tanh(x, name='%s' %(name))
        elif("relu" == activation):
            return tf.nn.relu(x, name='%s' %(name))
        elif("lrelu" == activation):
            return tf.nn.leaky_relu(x, name='%s' %(name))
        elif("elu" == activation):
            return tf.nn.elu(x, name='%s' %(name))
        elif("swish" == activation):
            return tf.nn.swish(x, name='%s' %(name))
        else: return x

    def dropout(self, x, rate=0.5, name=''):

        y = tf.nn.dropout(x=x, rate=rate, name=name)

        return y

    def batch_normalization(self, x, trainable=True, name='', verbose=True):

        # https://arxiv.org/pdf/1502.03167.pdf
        mean, variance = tf.nn.moments(x=x, axes=[0], keepdims=True, name="%s_mmt" %(name))

        c_in = x.get_shape().as_list()[-1]
        offset = self.get_variable(shape=[c_in], constant=0, \
            trainable=trainable, name="%s_ofs" %(name))
        scale = self.get_variable(shape=[c_in], constant=1, \
            trainable=trainable, name="%s_sce" %(name))

        y = tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=offset,
            scale=scale,
            variance_epsilon=1e-12,
            name=name
        )

        if(verbose): print("BN (%s)" %(name), x.shape, "->", y.shape)
        return y

    def layer_normalization(self, x, trainable=True, name='', verbose=True):

        len_xdim = len(x.shape)
        if(len_xdim == 2): x = tf.transpose(x, [1, 0])
        elif(len_xdim == 3): x = tf.transpose(x, [2, 1, 0])
        elif(len_xdim == 4): x = tf.transpose(x, [3, 1, 2, 0])
        elif(len_xdim == 5): x = tf.transpose(x, [4, 1, 2, 3, 0])

        mean, variance = tf.nn.moments(x=x, axes=[0], keepdims=True, name="%s_mmt" %(name))

        c_in = x.get_shape().as_list()[-1]
        offset = self.get_variable(shape=[c_in], constant=0, \
            trainable=trainable, name="%s_ofs" %(name))
        scale = self.get_variable(shape=[c_in], constant=1, \
            trainable=trainable, name="%s_sce" %(name))

        y = tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=offset,
            scale=scale,
            variance_epsilon=1e-12,
            name=name
        )

        if(len_xdim == 2): y = tf.transpose(y, [1, 0])
        elif(len_xdim == 3): y = tf.transpose(y, [2, 1, 0])
        elif(len_xdim == 4): y = tf.transpose(y, [3, 1, 2, 0])
        elif(len_xdim == 5): y = tf.transpose(y, [4, 1, 2, 3, 0])

        if(verbose): print("LN (%s)" %(name), x.shape, "->", y.shape)
        return y

    def maxpool(self, x, ksize=2, strides=1, \
        padding='SAME', name='', verbose=True):

        if(len(x.shape) == 3):
            y = tf.nn.max_pool1d(
                input=x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format='NWC',
                name=name
            )
        elif(len(x.shape) == 4):
            y = tf.nn.max_pool2d(
                input=x,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format='NHWC',
                name=name
            )

        if(verbose): print("MaxPool (%s)" %(name), x.shape, "->", y.shape)
        return y

    def conv1d(self, x, stride, \
        filter_size=[3, 16, 32], dilations=[1, 1, 1], \
        padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

        w = self.get_variable(shape=filter_size, \
            trainable=trainable, name='%s_w' %(name))
        if(usebias): b = self.get_variable(shape=[filter_size[-1]], \
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

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=trainable, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def convt1d(self, x, stride, output_shape, \
        filter_size=[3, 16, 32], dilations=[1, 1, 1], \
        padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

        w = self.get_variable(shape=filter_size, \
            trainable=trainable, name='%s_w' %(name))
        if(usebias): b = self.get_variable(shape=[filter_size[-2]], \
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

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=trainable, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def conv2d(self, x, stride, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

        w = self.get_variable(shape=filter_size, \
            trainable=trainable, name='%s_w' %(name))
        if(usebias): b = self.get_variable(shape=[filter_size[-1]], \
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

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=trainable, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def convt2d(self, x, stride, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

        w = self.get_variable(shape=filter_size, \
            trainable=trainable, name='%s_w' %(name))
        if(usebias): b = self.get_variable(shape=[filter_size[-2]], \
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

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=trainable, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def fully_connected(self, x, c_out, \
        batch_norm=False, trainable=True, activation=None, usebias=True, name='', verbose=True):

        c_in, c_out = x.get_shape().as_list()[-1], int(c_out)

        w = self.get_variable(shape=[c_in, c_out], \
            trainable=trainable, name='%s_w' %(name))
        if(usebias): b = self.get_variable(shape=[c_out], \
            trainable=trainable, name='%s_b' %(name))

        wx = tf.linalg.matmul(x, w, name='%s_mul' %(name))
        if(usebias): y = tf.math.add(wx, b, name='%s_add' %(name))
        else: y = wx
        if(verbose): print("FC (%s)" %(name), x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=trainable, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)
