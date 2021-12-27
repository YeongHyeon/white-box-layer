import numpy as np
import tensorflow as tf

def embedding(layer, x, dim_model, name='emb', verbose=True):

    emb = layer.fully_connected(x=x, c_out=dim_model, \
        batch_norm=False, activation=None, name="%s" %(name), verbose=verbose)

    return emb

def feed_forward_network(layer, x, dim_ff, dim_model, name='ffn', verbose=True):

    ff1 = layer.fully_connected(x=x, c_out=dim_ff, \
        batch_norm=False, activation='relu', name="%s_0" %(name), verbose=verbose)
    ff2 = layer.fully_connected(x=ff1, c_out=dim_model, \
        batch_norm=False, activation=None, name="%s_1" %(name), verbose=verbose)

    return ff2

def get_angles(pos, i, dim_model):
    # https://www.tensorflow.org/text/tutorials/transformer?hl=en

    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim_model))

    return pos * angle_rates

def positional_encoding(position, dim_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(dim_model)[np.newaxis, :],
                          dim_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def concat_heads(x, verbose=True):
    # https://www.tensorflow.org/text/tutorials/transformer?hl=en

    [d_n, d_s, d_h, d_fh] = x.shape
    xc = tf.reshape(x, (d_n, d_s, -1))

    if(verbose): print("Concat Head", x.shape, "->", xc.shape)
    return xc

def self_attention(layer, x_query, x_key, x_value, num_head=1, mask_idx=-1, udmask=False, name='enc', verbose=True):

    [_, d_s, d_f] = x_query.shape

    enc_query = layer.fully_connected(x=x_query, c_out=d_f, \
        batch_norm=False, activation=None, name="%s-query" %(name), verbose=verbose)
    enc_key = layer.fully_connected(x=x_key, c_out=d_f, \
        batch_norm=False, activation=None, name="%s-key" %(name), verbose=verbose)
    enc_value = layer.fully_connected(x=x_value, c_out=d_f, \
        batch_norm=False, activation=None, name="%s-value" %(name), verbose=verbose)

    sq_dk = tf.math.sqrt(float(d_f))
    enc_qk = []
    if(num_head != 1):
        list_query = tf.split(enc_query, num_or_size_splits=num_head, axis=2)
        list_key = tf.split(enc_key, num_or_size_splits=num_head, axis=2)
        list_value = tf.split(enc_value, num_or_size_splits=num_head, axis=2)

        for idx_query, _ in enumerate(list_query):
            enc_qk.append(tf.matmul(a=list_query[idx_query], b=list_key[idx_query], transpose_a=False, transpose_b=True) / sq_dk)

        enc_qk = tf.stack(enc_qk)
    else:
        enc_qk = tf.matmul(a=enc_query, b=enc_key, transpose_a=False, transpose_b=True) / sq_dk

    if(udmask): # upper diagonal masking
        enc_qk = tf.where(tf.linalg.band_part(enc_qk, -1, 0)==0, -1e+9, enc_qk)
    enc_smax_qk = tf.nn.softmax(enc_qk, axis=-1)

    if(num_head != 1):
        enc_qkv = []
        for idx_value, _ in enumerate(list_value):
            enc_qkv.append(tf.matmul(enc_smax_qk[idx_value], list_value[idx_value]))
        enc_qkv = tf.transpose(tf.stack(enc_qkv), [1, 2, 0, 3])
        enc_qkv = concat_heads(x=enc_qkv, verbose=verbose)
    else:
        enc_qkv = tf.matmul(enc_smax_qk, enc_value)

    if(verbose): print("Self-Attn (Head: %d)" %(num_head), x_query.shape, "->", enc_qkv.shape)
    return {'query':enc_query, 'key':enc_key, 'value':enc_value, 'attention':enc_smax_qk, 'output':enc_qkv}
