# coding=utf-8

import tensorflow as tf


class DeepCoNN(object):

    def __init__(self, user_length, item_length, num_classes, user_vocab_size, item_vocab_size,
                 fm_k, n_latent, user_num, item_num,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, l2_reg_V=0.0):

        # Parameters:
        #   user_length: max_length of user review text
        #   item_length: max_length of item review text
        #   num_classes: review rating categories
        #   user_vocab_size: word vocabulary size of user review text
        #   item vocab size: word vocabulary size of item review text
        #   fm_k : Hyper parameter k of factorization machine
        #   n_latent: hidden size of the shared layer
        #   user_num: number of users in all samples
        #   item_num: number of items in all samples
        #   embedding_size: embedding size of words
        #   filter_sizes: filter sizes of CNN
        #   number_filters: number of filters of CNN
        #   l2_reg_lambda: l2 regularization weights
        #   l2_reg_V: not used parameters

        self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W"
            )
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W"
            )
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        # CNN for user review text
        pooled_outputs_u = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv"
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, user_length - filter_size + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool"
                )
                pooled_outputs_u.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(pooled_outputs_u, 3)
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, num_filters_total])

        # CNN for item review text
        pooled_outputs_i = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv"
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, item_length - filter_size+ 1, 1,1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name="pool"
                )
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, 3)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)

        # hidden layer
        with tf.name_scope("get_fea"):
            Wu = tf.get_variable(
                "Wu",
                shape=[num_filters_total, n_latent],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_fea = tf.matmul(self.h_drop_u, Wu) + bu

            Wi = tf.get_variable(
                "Wi",
                shape=[num_filters_total, n_latent],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_fea = tf.matmul(self.h_drop_i, Wi) + bi

        # fm layer
        with tf.name_scope("fm"):
            self.z = tf.nn.relu(tf.concat([self.u_fea, self.i_fea], axis=1))

            WF1 = tf.Variable(
                tf.random_uniform([n_latent * 2, 1], -0.1, 0.1),
                name="fm1"
            )
            WF2 = tf.Variable(
                tf.random_uniform([n_latent * 2, fm_k], -0.1, 0.1),
                name="fm2"
            )

            # the first order interactions
            one = tf.matmul(self.z, WF1)

            # the second order interactions
            inte1 = tf.matmul(self.z, WF2)
            inte2 = tf.matmul(tf.square(self.z), tf.square(WF2))

            inter = (tf.square(inte1) - inte2) * 0.5
            inter = tf.nn.dropout(inter, self.dropout_keep_prob)
            inter = tf.reduce_sum(inter, 1, keep_dims=True)

            # global bias
            b = tf.Variable(tf.constant(0.1), name="bias")

            # sum up
            self.predictions = one + inter + b

        with tf.name_scope("loss"):
            # lossed = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))
            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))