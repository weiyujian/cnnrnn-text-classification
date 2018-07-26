import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

class TextRNNCNN(object):
    """
    A RNNCNN for text classification.
    Uses an embedding layer, followed by a bi-gru, convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, hidden_unit,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.real_len = tf.placeholder(tf.int32, [None], name="real_len")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        
        # bi-gru layer
        final_hidden_size = 2 * hidden_unit
        self.attn_obj = MLPAttentionLayer(final_hidden_size)
        with tf.name_scope("bi-gru") as scope:
            fw_cell = rnn_cell.GRUCell(hidden_unit, activation=tf.nn.relu)
            fw_cell = rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = rnn_cell.GRUCell(hidden_unit, activation=tf.nn.relu)
            bw_cell = rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            rnn_outputs, _ = rnn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_chars, sequence_length=tf.cast(self.real_len, tf.int64), dtype=tf.float32)
            rnn_outputs = tf.concat(rnn_outputs, 2)
        
        attn_rnn_outputs = self.attn_obj.get_output_for(rnn_outputs) 
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, final_hidden_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    attn_rnn_outputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

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
            self.correct_pred_num = tf.reduce_sum(tf.cast(correct_predictions, tf.int32), name="correct_pred_num")

class MLPAttentionLayer():
    def __init__(self, num_units):
        self.num_units = num_units
        self.w0 = tf.get_variable(shape=[self.num_units, self.num_units], initializer=tf.random_normal_initializer(stddev=0.1), name="W0_mlp")
        self.w1 = tf.get_variable(shape=[self.num_units], initializer=tf.random_normal_initializer(stddev=0.1), name="W1_mlp")
        self.wb = tf.get_variable(shape=[self.num_units], initializer=tf.constant_initializer(0.1), name="Wb0_mlp")
          
    def get_output_for(self, inputs):
        # inputs : batch_size * seq_len * hidden_size
        sequence_length = inputs.get_shape()[1].value
        inputs_ = tf.reshape(inputs, [-1, self.num_units])
        hidden_ = tf.nn.tanh(tf.matmul(inputs_, self.w0) + self.wb)
        hidden_reshape = tf.reshape(hidden_, [-1, sequence_length, self.num_units])
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_reshape, self.w1), 2))
        alpha = tf.expand_dims(alpha, 2)
        outputs = tf.expand_dims(tf.multiply(alpha, inputs), -1)
        return outputs
    
    def get_output_for2(self, inputs):
        # no attention
        outputs = tf.expand_dims(inputs, -1)
        return outputs
