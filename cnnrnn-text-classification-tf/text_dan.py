import tensorflow as tf
import numpy as np
from util import highway,linear
import pdb

class TextDAN(object):
    """
    A FastText for text classification.
    Uses an embedding layer, followed by a average operation, and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.use_region_emb = True
        self.hidden_size = 512
        self.hidden_layer = 2
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            if self.use_region_emb:
                self.region_size = 5
                self.region_radius = self.region_size / 2
                self.k_matrix_embedding = tf.Variable(tf.random_uniform([vocab_size, self.region_size, embedding_size], -1.0, 1.0), name="k_matrix")
                self.embedded_chars = self.region_embedding(self.input_x)
                sequence_length = int(self.embedded_chars.shape[1])
            else:
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.h_flat_0 = tf.reduce_mean(self.embedded_chars, 1)

        # Fully Connected Layer
        with tf.name_scope("fuuly-connection"):
            for i in range(self.hidden_layer):
                exec("self.h_flat_%s = tf.nn.relu(linear(self.h_flat_%s, self.hidden_size, bias=True, scope='fc_%s'))" % (i+1,i,i+1))
        
        # Highway Layer
        exec("self.highway = highway(self.h_flat_%s, self.h_flat_%s.get_shape()[1], num_layers=1, bias=-0.5, scope='Highway')" % (self.hidden_layer,self.hidden_layer))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.hidden_size, num_classes],
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
            self.correct_pred_num = tf.reduce_sum(tf.cast(correct_predictions, tf.int32), name="correct_pred_num")
    
    def get_seq(self, inputs):
        neighbor_seq = map(lambda i: tf.slice(inputs, [0, i-self.region_radius], [-1, self.region_size]), xrange(self.region_radius, inputs.shape[1] - self.region_radius))
        neighbor_seq = tf.convert_to_tensor(neighbor_seq)
        neighbor_seq = tf.transpose(neighbor_seq, [1,0,2])
        return neighbor_seq
    
    def get_seq_without_loss(self, inputs):
        neighbor_seq = map(lambda i: tf.slice(inputs, [0, i-self.region_radius], [-1, self.region_size]), xrange(self.region_radius, inputs.shape[1] - self.region_radius))
        neighbor_begin = map(lambda i: tf.slice(inputs, [0, 0], [-1, self.region_size]), xrange(0, self.region_radius))
        neighbor_end = map(lambda i: tf.slice(inputs, [0, inputs.shape[1] - self.region_size], [-1, self.region_size]), xrange(0, self.region_radius))
        neighbor_seq = tf.concat([neighbor_begin, neighbor_seq, neighbor_end], 0)
        neighbor_seq = tf.convert_to_tensor(neighbor_seq)
        neighbor_seq = tf.transpose(neighbor_seq, [1,0,2])
        return neighbor_seq

    def region_embedding(self, inputs):
        region_k_seq = self.get_seq(inputs)
        region_k_emb = tf.nn.embedding_lookup(self.W, region_k_seq)
        trimed_inputs = inputs[:, self.region_radius: inputs.get_shape()[1] - self.region_radius]
        context_unit = tf.nn.embedding_lookup(self.k_matrix_embedding, trimed_inputs)
        projected_emb = region_k_emb * context_unit
        embedded_chars = tf.reduce_max(projected_emb, axis=2)
        return embedded_chars
