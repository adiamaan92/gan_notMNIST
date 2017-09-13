import tensorflow as tf
import numpy as np
import os
import Helper as hp
import matplotlib.pyplot as plt


def layer(input_node, output_node, name):
    """
    :param input_node: Input node for weights
    :param output_node: Output node for weights and bias
    :param name: Name of the layer. Useful during TensorBoard visualization
    :return: None
    """
    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight', shape=[input_node, output_node],
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name="bias", initializer=tf.zeros(shape=[output_node]))
    return weight, bias


def sample_z(m, n):
    """
    :param m: Dimension of the noise
    :param n: Number of samples from noise
    :return: Returns 2-Dimensional random number m*n of uniform distribution [-1, 1]
    """
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    """
    :param z: Noise that's mapped to the images
    :return: 784 Dimension tensor and a list of generator parameters theta_d
    """
    with tf.name_scope("generator"):
        g_w1, g_b1 = layer(100, 128, "hidden")
        g_w2, g_b2 = layer(128, 784, "output")
        list_histogram({"g_w1": g_w1, "g_b1": g_b1, "g_w2": g_w2, "g_b2": g_b2})

        g_h1 = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
        g_logit = tf.matmul(g_h1, g_w2) + g_b2
        g_prob = tf.nn.sigmoid(g_logit)
    return g_prob, [g_w1, g_b1, g_w2, g_b2]


def list_histogram(variables):
    """
    :param variables: Log the variables for TensorBoard visualization
    :return:
    """
    for key, var in variables.iteritems():
        tf.summary.histogram(key, var)


def discriminator(x):
    """
    :param x: Accepts flattened image from original data as well as simulated data from generator
    :return: Returns the probability for original image. 1 - Original, 0 - Counterfeit. Also returns
    discriminator parameters, theta_d
    """
    with tf.variable_scope("discriminator"):
        d_w1, d_b1 = layer(784, 128, "hidden")
        d_w2, d_b2 = layer(128, 1, "output")
        list_histogram({"d_w1": d_w1, "d_b1": d_b1, "d_w2": d_w2, "d_b2": d_b2})

        d_h1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
        d_logit = tf.matmul(d_h1, d_w2) + d_b2
        d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit, [d_w1, d_b1, d_w2, d_b2]


if __name__ == '__main__':

    # Declaring Parameters
    Z_dim = 100  # Noise dimension
    image_size = 28  # Image dimension. Height and Weight
    train_file = '../notMNIST_new.pickle' # Train File
    mini_batch = 128  # Mini batch size
    output = 'nonmnist'  # Output file to write the image output

    # Saving output images
    if not os.path.exists(output+'/'):
        os.makedirs(output+'/')

    # Reading Data
    train_dataset = hp.load_pickle(train_file=train_file, element_list=['train_dataset'])[0]
    train_dataset = hp.reformat(dataset=train_dataset, image_size=image_size)

    # TensorFlow Computational Graph declaration

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, shape=[None, 784])
        Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
        tf.summary.image('input', tf.reshape(X, [-1, 28, 28, 1]), 10)

    with tf.variable_scope("gan") as gan:
        g_sample, theta_g = generator(Z)
        d_real, d_logit_real, _ = discriminator(X)
        gan.reuse_variables()
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

    with tf.name_scope('train'):
        d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=theta_d)
        g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)

    # Merging all summaries to a single op
    merged = tf.summary.merge_all()

    i = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('train/', sess.graph)
        for it in range(10000):
            if it % 1000 == 0:
                samples = sess.run(g_sample, feed_dict={Z: sample_z(16, Z_dim)})

                fig = hp.plot(samples)
                plt.savefig(output+'/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)

            random_index = np.random.choice(range(train_dataset.shape[0]), mini_batch)
            X_mb = train_dataset[random_index]
            batch_data = train_dataset[random_index]

            summary, _, D_loss_curr = sess.run([merged, d_solver, d_loss],
                                               feed_dict={X: X_mb, Z: sample_z(mini_batch, Z_dim)})
            _, G_loss_curr = sess.run([g_solver, g_loss], feed_dict={Z: sample_z(mini_batch, Z_dim)})

            # Writing all summaries to file
            train_writer.add_summary(summary, it)
            train_writer.close()

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
