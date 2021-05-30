import whiteboxlayer.layers as wbl
import tensorflow as tf

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = kwargs['who_am_i']
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.filters = kwargs['filters']

        self.layer = wbl.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, verbose=False):

        logit = self.__nn(x=x, name=self.who_am_i, verbose=verbose)
        y_hat = tf.nn.softmax(logit, name="y_hat")

        return logit, y_hat

    def __nn(self, x, name='neuralnet', verbose=True):

        for idx, _ in enumerate(self.filters[:-1]):
            if(idx == 0): continue
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[3, 3, self.filters[idx-1], self.filters[idx]], \
                activation='relu', name='%s-%dconv' %(name, idx), verbose=verbose)
            x = self.layer.maxpool(x=x, ksize=2, strides=2, \
                name='%s-%dmp' %(name, idx), verbose=verbose)

        x = tf.reshape(x, shape=[x.shape[0], -1], name="flat")
        x = self.layer.fully_connected(x=x, c_out=self.filters[-1], \
                activation='relu', name="%s-clf0" %(name), verbose=verbose)
        x = self.layer.fully_connected(x=x, c_out=self.num_class, \
                activation=None, name="%s-clf1" %(name), verbose=verbose)

        return x

model = Neuralnet(\
    who_am_i="CNN", \
    dim_h=28, dim_w=28, dim_c=1, \
    num_class=10, \
    filters=[1, 32, 64, 128])

dummy = tf.zeros((1, model.dim_h, model.dim_w, model.dim_c), dtype=tf.float32)
model.forward(x=dummy, verbose=True)
