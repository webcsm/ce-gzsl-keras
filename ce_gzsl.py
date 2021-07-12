# -*- coding: utf-8 -*-

import os
import uuid
import time
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, ReLU
import numpy as np
from dataset import Dataset
from classifier import Classifier
from util import map_label
import tensorflow.keras.backend as K
import logging


tf.compat.v1.disable_eager_execution()

logging.basicConfig(level=logging.INFO)


class CE_GZSL:

    # optimizers

    generator_optimizer = None
    discriminator_optimizer = None

    # nets

    embedding_net = None
    comparator_net = None
    generator_net = None
    discriminator_net = None

    # params

    discriminator_iterations = None
    generator_noise = None
    gp_weight = None
    instance_weight = None
    class_weight = None
    instance_temperature = None
    class_temperature = None
    synthetic_number = None
    gzsl = None
    visual_size = None

    def __init__(self, generator_optimizer: keras.optimizers.Optimizer,
                 discriminator_optimizer: keras.optimizers.Optimizer,
                 args: dict, **kwargs):
        super(CE_GZSL, self).__init__(**kwargs)

        self.embedding(args["visual_size"], args["embedding_hidden"], args["embedding_size"])
        self.comparator(args["embedding_hidden"], args["attribute_size"], args["comparator_hidden"])
        self.generator(args["visual_size"], args["attribute_size"], args["generator_noise"], args["generator_hidden"])
        self.discriminator(args["visual_size"], args["attribute_size"], args["discriminator_hidden"])

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.instance_weight = args["instance_weight"]
        self.class_weight = args["class_weight"]
        self.instance_temperature = args["instance_temperature"]
        self.class_temperature = args["class_temperature"]
        self.gp_weight = args["gp_weight"]
        self.synthetic_number = args["synthetic_number"]
        self.gzsl = args["gzsl"]
        self.visual_size = args["visual_size"]

        self.discriminator_iterations = args["discriminator_iterations"]
        self.generator_noise = args["generator_noise"]

    def summary(self):

        networks = [self.embedding_net, self.comparator_net, self.generator_net, self.discriminator_net]

        for net in networks:
            net.summary()

    def embedding(self, visual_size, hidden_units, embedded_size):

        inputs = keras.Input(shape=visual_size)
        x = keras.layers.Dense(hidden_units)(inputs)
        embed_h = ReLU(name="embed_h")(x)
        x = keras.layers.Dense(embedded_size)(embed_h)

        embed_z = keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1), name="embed_z")(x)

        self.embedding_net = keras.Model(inputs, [embed_h, embed_z], name="embedding")

    def comparator(self, embedding_size, attribute_size, hidden_units):

        inputs = keras.Input(shape=embedding_size + attribute_size)
        x = keras.layers.Dense(hidden_units)(inputs)
        x = LeakyReLU(0.2)(x)
        output = keras.layers.Dense(1, name="comp_out")(x)

        self.comparator_net = keras.Model(inputs, output, name="comparator")

    def generator(self, visual_size, attribute_size, noise, hidden_units):

        inputs = keras.Input(shape=attribute_size + noise)
        x = keras.layers.Dense(hidden_units)(inputs)
        x = LeakyReLU(0.2)(x)
        x = keras.layers.Dense(visual_size)(x)
        output = ReLU(name="gen_out")(x)
        self.generator_net = keras.Model(inputs, output, name="generator")

    def discriminator(self, visual_size, attribute_size, hidden_units):

        inputs = keras.Input(shape=visual_size + attribute_size)
        x = keras.layers.Dense(hidden_units)(inputs)
        x = LeakyReLU(0.2)(x)
        output = keras.layers.Dense(1, name="disc_out")(x)
        self.discriminator_net = keras.Model(inputs, output, name="discriminator")

    def d_loss_fn(self, real_logits, fake_logits):

        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return fake_loss - real_loss

    def gradient_penalty(self, batch_size, real_images, fake_images, attribute_data):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform(shape=(batch_size, 1))
        alpha = tf.tile(alpha, (1, real_images.shape[1]))
        interpolated = real_images * alpha + (1-alpha) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator_net(tf.concat([interpolated, attribute_data], axis=1))

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def contrastive_criterion(self, labels, feature_vectors):

        # Compute logits
        anchor_dot_contrast = tf.divide(
            tf.matmul(
                feature_vectors, tf.transpose(feature_vectors)
            ),
            self.instance_temperature,
        )

        logits_max = tf.reduce_max(anchor_dot_contrast, 1, keepdims=True)
        logits = anchor_dot_contrast - logits_max

        # Expand to [batch_size, 1]
        labels = tf.reshape(labels, (-1, 1))
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), dtype=tf.float32)

        # rosife: all except anchor
        logits_mask = tf.Variable(tf.ones_like(mask))
        indices = tf.reshape(tf.range(0, tf.shape(mask)[0]), (-1, 1))
        indices = tf.concat([indices, indices], axis=1)
        updates = tf.zeros((tf.shape(mask)[0]))
        logits_mask.scatter_nd_update(indices, updates)

        # rosife: positive except anchor
        mask = mask * logits_mask
        single_samples = tf.cast(tf.equal(tf.reduce_sum(mask, axis=1), 0), dtype=tf.float32)

        # compute log_prob
        masked_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(masked_logits, 1, keepdims=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, 1) / (tf.reduce_sum(mask, 1)+single_samples)

        # loss
        loss = -mean_log_prob_pos * (1 - single_samples)
        loss = tf.reduce_sum(loss) / (tf.cast(tf.shape(loss)[0], dtype=tf.float32) - tf.reduce_sum(single_samples))

        return loss

    def class_scores_for_loop(self, embed, input_label, attribute_seen):

        n_class_seen = attribute_seen.shape[0]

        expand_embed = tf.reshape(tf.tile(tf.expand_dims(embed, 1), [1, n_class_seen, 1]), [embed.shape[0] * n_class_seen, -1])
        expand_att = tf.reshape(tf.tile(tf.expand_dims(attribute_seen, 0), [embed.shape[0], 1, 1]), [embed.shape[0] * n_class_seen, -1])
        all_scores = tf.reshape(tf.divide(self.comparator_net(tf.concat([expand_embed, expand_att], axis=1)), self.class_temperature), [embed.shape[0], n_class_seen])

        score_max = tf.reduce_max(all_scores, axis=1, keepdims=True)
        # normalize the scores for stable training
        scores_norm = all_scores - score_max

        exp_scores = tf.exp(scores_norm)

        mask = tf.one_hot(input_label, n_class_seen)

        log_scores = scores_norm - tf.math.log(tf.reduce_sum(exp_scores, axis=1, keepdims=True))

        cls_loss = -tf.reduce_mean(tf.reduce_sum(mask * log_scores, axis=1) / tf.reduce_sum(mask, axis=1))

        return cls_loss

    def generate_synthetic_features(self, classes, attribute_data):

        nclass = classes.shape[0]
        syn_feature = tf.Variable(tf.zeros((self.synthetic_number * nclass, self.visual_size)))
        syn_label = tf.Variable(tf.zeros((self.synthetic_number * nclass, 1), dtype=classes.dtype))

        for i in range(nclass):

            iclass = classes[i]
            iclass_att = attribute_data[iclass]

            syn_att = tf.repeat(tf.reshape(iclass_att, (1, -1)), self.synthetic_number, axis=0)

            syn_noise = tf.random.normal(shape=(self.synthetic_number, self.generator_noise))

            output = self.generator_net(tf.concat([syn_noise, syn_att], axis=1))

            syn_feature[i * self.synthetic_number:(i+1) * self.synthetic_number, :].assign(output)
            syn_label[i * self.synthetic_number:(i+1) * self.synthetic_number, :].assign(tf.fill((self.synthetic_number, 1), iclass))

        return syn_feature, syn_label

    def train_step(self, real_features, attribute_data, labels, attribute_seen):

        batch_size = tf.shape(real_features)[0]

        d_loss_tracker = keras.metrics.Mean()

        for i in range(self.discriminator_iterations):

            # Get the latent vector
            noise_data = tf.random.normal(shape=(batch_size, self.generator_noise))

            with tf.GradientTape() as tape:

                embed_real, z_real = self.embedding_net(real_features)

                real_ins_contras_loss = self.contrastive_criterion(labels, z_real)
                cls_loss_real = self.class_scores_for_loop(embed_real, labels, attribute_seen)

                # Generate fake images from the latent vector
                fake_features = self.generator_net(tf.concat([noise_data, attribute_data], axis=1))
                # Get the logits for the fake images
                fake_logits = self.discriminator_net(tf.concat([fake_features, attribute_data], axis=1))
                # Get the logits for the real images
                real_logits = self.discriminator_net(tf.concat([real_features, attribute_data], axis=1))

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_logits, fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_features, fake_features, attribute_data)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight + real_ins_contras_loss + cls_loss_real

            trainable_variables = self.discriminator_net.trainable_variables + self.embedding_net.trainable_variables + self.comparator_net.trainable_variables

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, trainable_variables)

            # Update the weights of the discriminator using the discriminator optimizer
            self.discriminator_optimizer.apply_gradients(zip(d_gradient, trainable_variables))

            d_loss_tracker.update_state(d_loss)

        # Train the generator
        # Get the latent vector
        noise_data = tf.random.normal(shape=(batch_size, self.generator_noise))

        with tf.GradientTape() as tape:

            embed_real, z_real = self.embedding_net(real_features)

            embed_fake, z_fake = self.embedding_net(fake_features)

            fake_ins_contras_loss = self.contrastive_criterion(tf.concat([labels, labels], axis=0), tf.concat([z_fake, z_real], axis=0))
            cls_loss_fake = self.class_scores_for_loop(embed_fake, labels, attribute_seen)

            # Generate fake images using the generator
            fake_features = self.generator_net(tf.concat([noise_data, attribute_data], axis=1))
            # Get the discriminator logits for fake images
            fake_logits = self.discriminator_net(tf.concat([fake_features, attribute_data], axis=1))
            # Calculate the generator loss
            G_cost = -tf.reduce_mean(fake_logits)

            errG = G_cost + self.instance_weight * fake_ins_contras_loss + self.class_weight * cls_loss_fake

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(errG, self.generator_net.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator_net.trainable_variables))

        return d_loss_tracker.result(), errG, cls_loss_fake, cls_loss_real, fake_ins_contras_loss, real_ins_contras_loss

    def fit(self, dataset):

        train_features = dataset.train_features()
        train_attributes = dataset.train_attributes()
        train_labels = dataset.train_labels()

        unseen_classes = tf.constant(dataset.unseen_classes())
        seen_classes = tf.constant(dataset.seen_classes())

        attributes = tf.constant(dataset.attributes())

        train_feat_ds = tf.data.Dataset.from_tensor_slices(train_features)
        train_feat_ds = train_feat_ds.shuffle(buffer_size=train_features.shape[0], seed=seed).batch(batch_size)

        train_att_ds = tf.data.Dataset.from_tensor_slices(train_attributes)
        train_att_ds = train_att_ds.shuffle(buffer_size=train_attributes.shape[0], seed=seed).batch(batch_size)

        train_label_ds = tf.data.Dataset.from_tensor_slices(train_labels)
        train_label_ds = train_label_ds.shuffle(buffer_size=train_labels.shape[0], seed=seed).batch(batch_size)

        attribute_seen = tf.constant(ds.attribute_seen())

        for epoch in range(epochs):

            epoch_start = time.time()

            att_it = train_att_ds.__iter__()
            label_it = train_label_ds.__iter__()

            d_loss_tracker = keras.metrics.Mean()
            g_loss_tracker = keras.metrics.Mean()

            for step, train_feat in enumerate(train_feat_ds):

                train_att = att_it.next()
                train_label = label_it.next()

                train_label = map_label(train_label, seen_classes)

                d_loss, g_loss, cls_loss_fake, cls_loss_real, fake_ins_contras_loss, real_ins_contras_loss = ce_gzsl.train_step(train_feat, train_att, train_label, attribute_seen)

                d_loss_tracker.update_state(d_loss)
                g_loss_tracker.update_state(g_loss)

            logging.info("main epoch {} - d_loss {:.4f} - g_loss {:.4f} - time: {:.4f}".format(epoch, d_loss_tracker.result(), g_loss_tracker.result(), time.time() - epoch_start))

            # classification

            cls_start = time.time()

            if self.gzsl:

                syn_feature, syn_label = self.generate_synthetic_features(unseen_classes, attributes)

                train_x = tf.concat([train_features, syn_feature], axis=0)
                train_y = tf.concat([train_labels.reshape(-1, 1), syn_label], axis=0)
                num_classes = tf.size(unseen_classes) + tf.size(seen_classes)

                cls = Classifier(train_x, train_y, self.embedding_net, seed, num_classes, 25,
                                 self.synthetic_number, self.visual_size, cls_lr, beta1, beta2, dataset)

                acc_seen, acc_unseen, acc_h = cls.fit()

                logging.info('best acc: seen {:.4f} - unseen {:.4f} - H {:.4f} - time {:.4f}'.format(acc_seen, acc_unseen, acc_h, time.time() - cls_start))

            else:
                syn_feature, syn_label = self.generate_synthetic_features(unseen_classes, attributes)
                labels = map_label(syn_label, unseen_classes)
                num_classes = tf.size(unseen_classes)

                cls = Classifier(syn_feature, labels, self.embedding_net, seed, num_classes, 100,
                                 self.synthetic_number, self.visual_size, cls_lr, beta1, beta2, dataset)

                acc = cls.fit_zsl()

                logging.info('best acc: {:.4f} - time {:.4f}'.format(acc, time.time() - cls_start))

            if (epoch + 1) % checkpoint_epochs == 0:
                logging.info("saving checkpoint: {}".format(exp_path))
                np.save(os.path.join(exp_path, "syn_feature.npy"), syn_feature.numpy())
                np.save(os.path.join(exp_path, "syn_label.npy"), syn_label.numpy())
                self.generator_net.save(os.path.join(exp_path, "generator.h5"))
                self.discriminator_net.save(os.path.join(exp_path, "discriminator.h5"))
                self.comparator_net.save(os.path.join(exp_path, "comparator.h5"))
                self.embedding_net.save(os.path.join(exp_path, "embedding.h5"))


validation = False
preprocessing = False

exp_local_path = "<local_path>"
exp_remote_path = "<remote_path>"

exp_path = os.path.join(exp_remote_path, "cegzsl_experiments", str(uuid.uuid4()))
os.makedirs(exp_path)

data_local_path = "xlsa17"
data_remote_path = "xlsa17"

ds = Dataset(data_remote_path)
ds.read("APY", preprocessing=preprocessing, validation=validation)

args = {"visual_size": ds.feature_size(),
        "attribute_size": ds.attribute_size(),
        "embedding_size": 512,
        "embedding_hidden": 2048,
        "comparator_hidden": 2048,
        "generator_hidden": 4096,
        "generator_noise": 1024,
        "discriminator_hidden": 4096,
        "discriminator_iterations": 5,
        "instance_weight": 0.001,
        "class_weight": 0.001,
        "instance_temperature": 0.1,
        "class_temperature": 0.1,
        "gp_weight": 10.0,
        "gzsl": False,
        "synthetic_number": 100}

# main training

epochs = 2000
batch_size = 4096
learning_rate = 0.0001
lr_decay = 0.99
lr_decay_epochs = 100
beta1 = 0.5
beta2 = 0.999
seed = 1985
checkpoint_epochs = 100

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

steps_per_epoch = int(np.ceil(ds.train_features().shape[0] / batch_size))
lr_decay_steps = lr_decay_epochs * steps_per_epoch

# classifier

cls_lr = 0.001

gen_lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                              decay_steps=lr_decay_steps, decay_rate=lr_decay)

disc_lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                               decay_steps=lr_decay_steps, decay_rate=lr_decay)

gen_opt = keras.optimizers.Adam(learning_rate=gen_lr_schedule, beta_1=beta1, beta_2=beta2)
disc_opt = keras.optimizers.Adam(learning_rate=disc_lr_schedule, beta_1=beta1, beta_2=beta2)

ce_gzsl = CE_GZSL(gen_opt, disc_opt, args)

# ce_gzsl.summary()
ce_gzsl.fit(ds)
