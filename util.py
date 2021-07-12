# -*- coding: utf-8 -*-

import tensorflow as tf


def map_label(label, classes):

    mapped_label = tf.Variable(tf.zeros(label.shape, dtype=label.dtype))

    for c in range(classes.shape[0]):
        mask = tf.where(tf.equal(label, classes[c]))
        mapped_label.scatter_nd_update(mask, tf.fill(mask.shape[0], c))

    return mapped_label
