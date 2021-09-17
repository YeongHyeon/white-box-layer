import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

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

def get_flops(conc_func, path_out='flops.txt'):

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(conc_func)

    with tf.Graph().as_default() as graph:
        tf.compat.v1.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        flop_tot = flops.total_float_ops
        ftxt = open(path_out, "w")
        for idx, name in enumerate(['', 'K', 'M', 'G', 'T']):
            text = '%.3f [%sFLOPS]' %(flop_tot/10**(3*idx), name)
            print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

def save_params(model, optimizer, name='base', path_ckpt='.', tflite=False):

    if(tflite):
        # https://github.com/tensorflow/tensorflow/issues/42818
        converter = tf.lite.TFLiteConverter.from_concrete_functions([conc_func])

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()

        with open('model_%s.tflite' %(name), 'wb') as f:
            f.write(tflite_model)
    else:
        vars_to_save = model.layer.parameters.copy()
        vars_to_save["optimizer"] = optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(path_ckpt, 'model_%s' %(name)), max_to_keep=1)
        ckptman.save()

def load_params(model, optimizer, name='base', path_ckpt='.'):

    vars_to_load = model.layer.parameters.copy()
    vars_to_load["optimizer"] = optimizer

    ckpt = tf.train.Checkpoint(**vars_to_load)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join(path_ckpt, 'model_%s' %(name)))
    status = ckpt.restore(latest_ckpt)
    status.expect_partial()

    return model, optimizer
