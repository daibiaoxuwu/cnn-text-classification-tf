import tensorflow as tf
import numpy as np


class Cnnmodel(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
            # Convolution Layer
        filter_shape1 = [3, embedding_size, 1, 3]
        W1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W")
        b1 = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
        self.conv1 = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W1,
            strides=[1, 1, 1, 1],
            padding="VALID"
            )
        # Apply nonlinearity
        h1 = tf.nn.relu(tf.nn.bias_add(self.conv1, b1), name="relu")
        # Maxpooling over the outputs
        pooled1 = tf.nn.max_pool(
            h1,
            ksize=[1, sequence_length - 3 + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
            )
        pooled_outputs.append(pooled1)
        # Convolution Layer
        filter_shape2 = [3, embedding_size, 1, 3]
        W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W")
        b2 = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
        self.conv2 = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W2,
            strides=[1, 1, 1, 1],
            padding="VALID"
            )
        # Apply nonlinearity
        h2 = tf.nn.relu(tf.nn.bias_add(self.conv2, b2), name="relu")
        # Maxpooling over the outputs
        pooled2 = tf.nn.max_pool(
            h2,
            ksize=[1, sequence_length - 3 + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
            )
        pooled_outputs.append(pooled2)
        # Convolution Layer
        filter_shape3 = [3, embedding_size, 1, 3]
        W3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W")
        b3 = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
        self.conv3 = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W3,
            strides=[1, 1, 1, 1],
            padding="VALID"
            )
        # Apply nonlinearity
        h3 = tf.nn.relu(tf.nn.bias_add(self.conv3, b3), name="relu")
        # Maxpooling over the outputs
        pooled3 = tf.nn.max_pool(
            h3,
            ksize=[1, sequence_length - 3 + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled3)
        # Convolution Layer

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-3)
