
import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io
import pickle

def load_pickle(train_file, element_list):

    """
    :param train_file: Pickle file containing elements in a dictionary format
    :param element_list: the elements to be returned from the pickle file
    :return: returns the elements in the same order received
    """

    file_stream = file_io.FileIO(train_file, mode='r')
    save = pickle.load(file_stream)
    return [save[i] for i in element_list]


def reformat(dataset, image_size, labels=None, num_labels=None):

    """
    Returns data set and labels(if given) reformatted in a single dimension
    :param dataset: data_set containing all attributes (x)
    :param labels: labels of the data set (y) (optional)
    :param image_size: the size of the image size (width == height)
    :param num_labels: number of output categories (one hot encoding) (optional)
    :return: returns the dataset and labels squashed to a single dimension
    """

    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    if labels is not None:
        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    else:
        return dataset
    return dataset, labels




def variable_summaries(variables):
    for key, var in variables.iteritems():
        with tf.name_scope(key):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
