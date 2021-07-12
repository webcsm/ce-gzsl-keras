# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn.metrics import balanced_accuracy_score
from util import map_label
import warnings
import logging


logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)


class Classifier:

    classifier_net = None
    epochs = None
    batch_size = None
    embedding_net = None
    n_class = None
    optimizer = None

    train_x = None
    train_y = None
    test_unseen_feature = None
    test_unseen_label = None
    unseen_classes = None
    seen_classes = None

    seed = None

    def __init__(self, train_x, train_y, embedding_net, seed, n_class, epochs, batch_size, embedding_size,
                 learning_rate, beta1, beta2, dataset):
        self.train_x = train_x
        self.train_y = train_y

        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.n_class = n_class
        self.embedding_net = embedding_net
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)

        self.classifier(embedding_size)

        self.test_unseen_feature = dataset.test_unseen_features()
        self.test_unseen_label = dataset.test_unseen_labels()
        self.test_seen_feature = dataset.test_seen_features()
        self.test_seen_label = dataset.test_seen_labels()
        self.unseen_classes = dataset.unseen_classes()
        self.seen_classes = dataset.seen_classes()

    def classifier(self, embedding_size):

        inputs = keras.Input(shape=embedding_size)
        output = keras.layers.Dense(self.n_class, name="output", activation='softmax')(inputs)
        self.classifier_net = keras.Model(inputs, output, name="classifier")

    def train_step(self, train_feat, train_label):

        with tf.GradientTape() as tape:

            embed, _ = self.embedding_net(train_feat)

            output = self.classifier_net(embed)

            one_hot_labels = tf.one_hot(train_label, self.n_class)

            loss = K.mean(K.categorical_crossentropy(tf.squeeze(one_hot_labels), output), axis=-1)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(loss, self.classifier_net.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.optimizer.apply_gradients(zip(gen_gradient, self.classifier_net.trainable_variables))

        return loss

    def fit_zsl(self):

        best_acc = 0

        train_feat_ds = tf.data.Dataset.from_tensor_slices(self.train_x)
        train_feat_ds = train_feat_ds.shuffle(buffer_size=self.train_x.shape[0], seed=self.seed).batch(self.batch_size)

        train_label_ds = tf.data.Dataset.from_tensor_slices(self.train_y)
        train_label_ds = train_label_ds.shuffle(buffer_size=self.train_y.shape[0], seed=self.seed).batch(self.batch_size)

        for epoch in range(self.epochs):

            label_it = train_label_ds.__iter__()

            loss_tracker = keras.metrics.Mean()

            for step, train_feat in enumerate(train_feat_ds):

                train_label = label_it.next()

                loss_tracker.update_state(self.train_step(train_feat, train_label))

            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseen_classes)

            if acc > best_acc:
                best_acc = acc

        return best_acc

    def val(self, features, labels, classes):

        predicted_label = tf.Variable(tf.zeros(labels.shape, dtype=labels.dtype))

        test_feat_ds = tf.data.Dataset.from_tensor_slices(features)
        test_feat_ds = test_feat_ds.batch(self.batch_size)

        for idx, _ in enumerate(test_feat_ds):

            i = idx * self.batch_size

            embed, _ = self.embedding_net(features[i:i+self.batch_size])
            output = self.classifier_net(embed)

            predicted_label[i:i+self.batch_size].assign(tf.argmax(output, axis=1, output_type=tf.int32))

        labels = map_label(labels, classes)

        acc = balanced_accuracy_score(labels.numpy(), predicted_label.numpy())

        return acc

    def fit(self):

        best_acc_seen = 0
        best_acc_unseen = 0
        best_acc_h = 0

        train_feat_ds = tf.data.Dataset.from_tensor_slices(self.train_x)
        train_feat_ds = train_feat_ds.shuffle(buffer_size=self.train_x.shape[0], seed=self.seed).batch(self.batch_size)

        train_label_ds = tf.data.Dataset.from_tensor_slices(self.train_y)
        train_label_ds = train_label_ds.shuffle(buffer_size=self.train_y.shape[0], seed=self.seed).batch(self.batch_size)

        for epoch in range(self.epochs):

            loss_tracker = keras.metrics.Mean()

            label_it = train_label_ds.__iter__()

            for step, train_feat in enumerate(train_feat_ds):

                train_label = label_it.next()

                loss_tracker.update_state(self.train_step(train_feat, train_label))

            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seen_classes)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseen_classes)

            acc_h = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)

            if acc_h > best_acc_h:
                best_acc_seen = acc_seen
                best_acc_unseen = acc_unseen
                best_acc_h = acc_h

        return best_acc_seen, best_acc_unseen, best_acc_h

    def val_gzsl(self, features, labels, classes):

        predicted_label = tf.Variable(tf.zeros(labels.shape, dtype=labels.dtype))

        test_feat_ds = tf.data.Dataset.from_tensor_slices(features)
        test_feat_ds = test_feat_ds.batch(self.batch_size)

        for idx, _ in enumerate(test_feat_ds):

            i = idx * self.batch_size

            embed, _ = self.embedding_net(features[i:i+self.batch_size])
            output = self.classifier_net(embed)

            predicted_label[i:i+self.batch_size].assign(tf.argmax(output, axis=1, output_type=tf.int32))

        # labels = map_label(labels, classes)

        acc = balanced_accuracy_score(labels.numpy(), predicted_label.numpy())

        return acc
