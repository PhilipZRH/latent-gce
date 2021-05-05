import tensorflow as tf


def image_encoder(input_tensor, latent_dim, name, reuse=False):
    kwargs = dict(strides=2, activation=tf.nn.relu, reuse=reuse)
    hidden = input_tensor
    hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs, name=name+'-conv1')
    hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs, name=name+'-conv2')
    hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs, name=name+'-conv3')
    hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs, name=name+'-conv4')
    hidden = tf.layers.flatten(hidden)
    assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
    hidden = tf.layers.dense(hidden, latent_dim, None, reuse=reuse, name=name+'-dense1')
    return hidden


def image_decoder(input_tensor, num_steps_observation, name, reuse=None):
    kwargs = dict(strides=2, activation=tf.nn.relu, reuse=reuse)
    hidden = tf.layers.dense(input_tensor, 1024, None, reuse=reuse, name=name+'-dense1')
    hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
    hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs, name=name+'-conv_t1')
    hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs, name=name+'-conv_t2')
    hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs, name=name+'-conv_t3')
    mean = tf.layers.conv2d_transpose(hidden, num_steps_observation * 3, 6, strides=2,
                                      reuse=reuse, name=name+'-conv_t4')
    assert mean.shape[1:].as_list() == [64, 64, num_steps_observation * 3], mean.shape
    return mean


def build_mlp(input_tensor,
              output_dim,
              name,
              hidden_dims=(1024, 1024, 1024),
              activation=tf.nn.relu,
              reuse=False):
    out = input_tensor
    i = 1
    for dim in hidden_dims:
        out = tf.layers.dense(out, dim, activation=activation, reuse=reuse, name=name+'-dense'+str(i))
        i += 1
    out = tf.layers.dense(out, output_dim, activation=None, reuse=reuse, name=name+'-dense'+str(i))
    return out
