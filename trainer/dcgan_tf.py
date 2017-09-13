import tensorflow as tf
import numpy as np
import os
import Helper as hp
import matplotlib.pyplot as plt
import math


def layer(shape,  name):
    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight', shape=shape,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name="bias", initializer=tf.zeros(shape=shape[-1]))
    return weight, bias


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        batch_size = tf.shape(input_)[0]
        output_newshape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_newshape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_newshape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def generator(z):
    with tf.name_scope("generator"):
        s_h, s_w = image_size, image_size
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

        fc_w1 = tf.get_variable(name="fc_w1", shape=[100, 7*7*64],
                                initializer=tf.contrib.layers.xavier_initializer())
        fc_b1 = tf.get_variable(name="fc_b1", initializer=tf.zeros(shape=[7*7*64]))

        fc1 = tf.nn.relu(tf.matmul(z, fc_w1) + fc_b1)

        fc1_reshape = tf.reshape(fc1, [-1, 7, 7, 64])

        h1, h1_w, h1_b = deconv2d(fc1_reshape, [-1, 14, 14, 32], name="g_h1", with_w=True)
        h1 = tf.nn.relu(h1)

        h2, h2_w, h2_b = deconv2d(h1, [-1, 28, 28, 1], name="g_h2", with_w=True)
        h2 = tf.nn.tanh(h2)
        # print(h2.shape)
        # h2 = tf.reshape(h2, [-1, 784])
        # print(h2.shape)

    return h2, [fc_w1, fc_b1, h1_w, h1_b, h2_w, h2_b]


def list_histogram(variables):
    for key, var in variables.iteritems():
        tf.summary.histogram(key, var)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        print(shape)
        matrix = tf.get_variable("Matrix", [4*4*64, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def discriminator(x):
    with tf.variable_scope("discriminator"):
        print(x.get_shape())
        h0 = lrelu(conv2d(x, 1, name="d_h0"))
        print(h0.get_shape())
        h1 = lrelu(conv2d(h0, 32, name="d_h1"))
        print(h1.get_shape())
        h2 = lrelu(conv2d(h1, 64, name="d_h2"))
        print(h2.get_shape())
        temp = tf.reshape(h2, [mini_batch, -1])
        print(temp.get_shape())
        h3, fc_w, fc_b = linear(tf.reshape(h2, [mini_batch, -1]), 1, 'd_h4_lin', with_w=True)

    return tf.nn.sigmoid(h3), h3, [fc_w, fc_b]


if __name__ == '__main__':

    # Declaring Parameters
    Z_dim = 100
    image_size = 28
    train_file = '../notMNIST_new.pickle'
    mini_batch = 128
    output = 'nonmnist'

    # Saving output images
    if not os.path.exists(output+'/'):
        os.makedirs(output+'/')

    # Reading Data
    train_dataset = hp.load_pickle(train_file=train_file, element_list=['train_dataset'])[0]
    train_dataset = hp.reformat(dataset=train_dataset, image_size=image_size)

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
        Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name="Z")
        X = tf.reshape(X, shape=[-1, 28, 28, 1])

    with tf.variable_scope("gan") as gan:
        g_sample, theta_g = generator(Z)
        d_real, d_logit_real, _ = discriminator(X)
        tf.get_variable_scope().reuse_variables()
        d_fake, d_logit_fake, theta_d = discriminator(g_sample)

    with tf.name_scope('discriminator_loss'):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real,
                                                                             labels=tf.ones_like(d_logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake,
                                                                             labels=tf.zeros_like(d_logit_fake)))
        d_loss = d_loss_real + d_loss_fake
        tf.summary.scalar('d_loss', d_loss)

    with tf.name_scope('generator_loss'):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake,
                                                                        labels=tf.ones_like(d_logit_fake)))
        tf.summary.scalar('g_loss', g_loss)

    with tf.name_scope('train') as scope:
        d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=theta_d)
        g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)

    merged = tf.summary.merge_all()

    i = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('train/', sess.graph)
        for it in range(10000):
            if it % 1000 == 0:
                mini_batch = 16
                samples = sess.run(g_sample, feed_dict={Z: sample_z(mini_batch, Z_dim)})

                fig = hp.plot(samples)
                plt.savefig(output+'/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)
                mini_batch = 128

            random_index = np.random.choice(range(train_dataset.shape[0]), mini_batch)
            X_mb = train_dataset[random_index]
            batch_data = train_dataset[random_index]

            print(X_mb.shape)

            summary, _, D_loss_curr = sess.run([merged, d_solver, d_loss],
                                               feed_dict={X: X_mb, Z: sample_z(mini_batch, Z_dim)})
            _, G_loss_curr = sess.run([g_solver, g_loss], feed_dict={Z: sample_z(mini_batch, Z_dim)})

            #train_writer.add_summary(summary, it)
            #train_writer.close()

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
