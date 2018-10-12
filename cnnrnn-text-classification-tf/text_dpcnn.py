import tensorflow as tf
import numpy as np
from util import highway
import pdb
class TextDPCNN(object):
    """
    A Deep Pyramid CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, num_blocks, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.filter_size = 3
        self.num_filters = num_filters
        self.use_region_emb = True
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
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        # 2.two layers of convs
        conv = self.dpcnn_two_layers_conv(self.embedded_chars_expanded)
        # 2.1 skip connection: add and activation
        b = tf.get_variable("b-inference", [self.num_filters])
        conv = tf.nn.relu(tf.nn.bias_add(conv, b), "relu-inference")
        conv = conv + self.embedded_chars_expanded
        # 3.repeat of building blocks
        for i in range(num_blocks):
            conv = self.dpcnn_pooling_two_conv(conv, i)
        # 4.max pooling
        seq_length1 = conv.get_shape().as_list()[1]
        seq_length2 = conv.get_shape().as_list()[2]
        pooling = tf.nn.max_pool(conv, ksize=[1, seq_length1, seq_length2, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool")
        fc_hidden_size = pooling.get_shape().as_list()[-1]
        self.h_pool_flat = tf.squeeze(pooling)
        # Fully Connected Layer
        with tf.name_scope("fc"):
            
            W_fc = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, fc_hidden_size],\
                stddev=0.1, dtype=tf.float32), name="W_fc")
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

    def dpcnn_pooling_two_conv(self, conv, layer_index):
        """
        pooling followed with two layers of conv, used by deep pyramid cnn.
        pooling-->conv-->conv-->skip connection
        conv:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
        return:[batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
        """
        with tf.variable_scope("pooling_two_conv_" + str(layer_index)):
            # 1. pooling:max-pooling with size 3 and stride 2==>reduce shape to half
            pooling = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',name="pool")
            # 2. two layer of conv
            conv = self.dpcnn_two_layers_conv(pooling)
            # 3. skip connection and activation
            b = tf.get_variable("b-poolcnn%s" % self.num_filters, [self.num_filters])
            conv = tf.nn.relu(tf.nn.bias_add(conv, b),"relu-poolcnn")
            conv = conv + pooling
        return conv

    def dpcnn_two_layers_conv(self, inputs):
        """
        two layers of conv
        inputs:[batch_size,total_sequence_length,embed_size,dimension]
        return:[batch_size,total_sequence_length,embed_size,num_filters]
        """
        #conv1
        channel = inputs.get_shape().as_list()[-1]
        hpcnn_number_filters = self.num_filters
        filter1 = tf.get_variable("filter1-%s" % self.filter_size,[self.filter_size, 1, channel, hpcnn_number_filters],initializer=tf.random_normal_initializer(stddev=0.1))
        conv1 = tf.nn.conv2d(inputs, filter1, strides=[1, 1, 1, 1], padding="SAME",name="conv1")
        conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
        b1 = tf.get_variable("b-cnn-%s" % hpcnn_number_filters, [hpcnn_number_filters])
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1),"relu1")
        # conv2
        filter2 = tf.get_variable("filter2-%s" % self.filter_size,[self.filter_size, 1, hpcnn_number_filters, hpcnn_number_filters],initializer=tf.random_normal_initializer(stddev=0.1))
        conv2 = tf.nn.conv2d(conv1, filter2, strides=[1, 1, 1, 1], padding="SAME",name="conv2")
        conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
        return conv2

