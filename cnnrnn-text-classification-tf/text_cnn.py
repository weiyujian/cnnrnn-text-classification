import tensorflow as tf
import numpy as np
from util import highway

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                conv_bn = tf.layers.batch_normalization(conv, training=self.is_training)
                # Apply nonlinearity
                #h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h = tf.nn.relu(conv_bn, name="relu")
                # Maxpooling over the outputs
                pool_size = sequence_length - filter_size + 1
                #pooled = self._chunk_max_pooling(h, topk)#sequence_length - filter_size + 1
                pooled = self._max_pooling(h, pool_size)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Fully Connected Layer
        with tf.name_scope("fc"):
            fc_hidden_size = num_filters_total             
            W_fc = tf.Variable(tf.truncated_normal(shape=[num_filters_total, fc_hidden_size],\
                stddev=0.1, dtype=tf.float32), name="W_fc")
            #b_fc = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b_fc")
            #self.fc = tf.nn.xw_plus_b(self.h_pool_flat, W_fc, b_fc)
            self.fc = tf.matmul(self.h_pool_flat, W_fc)
            self.fc_bn = tf.layers.batch_normalization(self.fc, training=self.is_training)
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")
        # Highway Layer
        self.highway = highway(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=-0.5, scope="Highway")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[fc_hidden_size, num_classes],
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
    
    def _max_pooling(self, inputs, filter_size):
        # max pooling
        pooled = tf.nn.max_pool(
            inputs,
            ksize=[1, filter_size, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        return pooled
    
    def _k_max_pooling(self, inputs, top_k):
        # k max pooling
        #inputs : batch_size, sequence_length, hidden_size, chanel_size]
        inputs = tf.transpose(inputs, [0,3,2,1]) # [batch_size, chanel_size, hidden_size, sequence_length]
        k_pooled = tf.nn.top_k(inputs, k=top_k, sorted=True, name='top_k')[0] # [batch_size, chanel_size, hidden_size, top_k]
        k_pooled = tf.transpose(k_pooled, [0,3,2,1]) #[batch_size, top_k, hidden_size, chanel_size]
        return k_pooled

    def _chunk_max_pooling(self, inputs, chunk_size):
        #chunk max pooling
        seq_len = inputs.get_shape()[1].values
        inputs_ = tf.split(inputs, chunk_size, axis=1) # seq_len/chunk_size list,element is  [batch_size, seq_len/chunk_size, hidden_size, chanel_size]
        chunk_pooled_list = []
        for i in range(len(inputs_)):
            chunk_ = inputs_[i]
            chunk_pool_ = self._max_pooling(chunk_, seq_len/chunk_size)
            chunk_pooled_list.append(chunk_pool_)
        chunk_pooled = tf.concat(chunk_pooled_list, axis=1)
        return chunk_pooled

