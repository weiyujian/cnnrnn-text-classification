import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import pdb
class TextRNN(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by a rnn and softmax layer.
    support gru lstm, and multi_rnn_cell
    support two ways attention
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, hidden_unit,
      embedding_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.real_len = tf.placeholder(tf.int32, [None], name="real_len")
        self.num_layers = 3 #whether use multi-rnn
        self.rnn_type = "lstm" # use lstm or gru
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        
        if self.rnn_type == "gru":
            # Create bi-gru layer and get the last output of real len
            with tf.name_scope("gru") as scope:
                fw_cell_0 = self.get_a_cell(hidden_unit)
                bw_cell_0 = self.get_a_cell(hidden_unit)
                if self.num_layers > 1:
                    fw_cell_1 = self.get_a_cell(hidden_unit)
                    bw_cell_1 = self.get_a_cell(hidden_unit)
                    fw_cell_0 = rnn_cell.MultiRNNCell([fw_cell_0] + [fw_cell_1] * (self.num_layers - 1))
                    bw_cell_0 = rnn_cell.MultiRNNCell([bw_cell_0] + [bw_cell_1] * (self.num_layers - 1))
                fw_cell = fw_cell_0
                bw_cell = bw_cell_0
                rnn_outputs, rnn_states = rnn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_chars, sequence_length=tf.cast(self.real_len, tf.int64), dtype=tf.float32)
        elif self.rnn_type == "lstm":
            #create lstm layer using dynamic rnn
            with tf.name_scope("lstm") as scope:
                fw_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_unit, forget_bias=0.0) for _ in xrange(self.num_layers)])
                bw_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_unit, forget_bias=0.0) for _ in xrange(self.num_layers)])
                rnn_outputs, nex_state = rnn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_chars,sequence_length=tf.cast(self.real_len, tf.int64), dtype=tf.float32)
        pdb.set_trace()
        #self.h_pool_flat = tf.concat(rnn_states, 1)
        rnn_outputs = tf.concat(rnn_outputs, 2)
        #self.h_pool_flat = self.get_last_output(rnn_outputs)
        self.attn_obj = MLPAttentionLayer(2 * hidden_unit)
        #self.attn_obj = DotProductAttentionLayer(2 * hidden_unit)
        self.h_pool_flat = self.attn_obj.get_output_for(rnn_outputs)
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[2 * hidden_unit, num_classes],
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
        
    def get_a_cell(self, num_units):
        cell = rnn_cell.GRUCell(num_units, activation=tf.nn.relu)
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        return cell

    def get_last_output(self, rnn_outputs):
        #get last output of both bw and fw for bi-rnn
        #rnn_outputs: batch * seq * h
        batch_size = tf.shape(rnn_outputs)[0]
        sequence_length = rnn_outputs.get_shape()[1].value
        num_units = rnn_outputs.get_shape()[2].value
        index = tf.range(0, batch_size) * sequence_length + (self.real_len - 1)
        #get the last output of fw and bw
        rnn_outputs_ = tf.gather(tf.reshape(rnn_outputs, [-1, num_units]), index)
        return rnn_outputs_

class DotProductAttentionLayer():
    def __init__(self, num_units):
        self.num_units = num_units
        self.w0 = tf.get_variable(shape=[self.num_units], initializer=tf.random_normal_initializer(stddev=0.1), name="W0")
           
    def get_output_for(self, inputs):
        # inputs : batch_size * seq_len * hidden_size
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(inputs, self.w0), 2))
        outputs = tf.reduce_sum(tf.multiply(inputs, tf.expand_dims(alpha,2)), 1)
        return outputs

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
        outputs = tf.reduce_sum(tf.multiply(alpha, inputs), 1)
        return outputs


