import tensorflow as tf
import numpy as np
from util import highway
import pdb

class TextAttnCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a multi-head attention, convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, num_heads, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
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
        
        ## Dropout
        self.embedded_chars_dropout = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob)

        #attention encoder
        with tf.name_scope("attention-encoder"):
            ### Multihead Attention
            emb_attn_encoder = self.multihead_attention(queries=self.embedded_chars_dropout,
                                    keys=self.embedded_chars_dropout,
                                    num_units=embedding_size,
                                    num_heads=num_heads,
                                    causality=False)
            ### Feed Forward
            emb_attn_encoder = self.feedforward(emb_attn_encoder, num_units=[4*embedding_size, embedding_size])
            emb_attn_encoder_expanded = tf.expand_dims(emb_attn_encoder, -1)
            
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    emb_attn_encoder_expanded,
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

    def multihead_attention(self, queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        causality=False,
                        scope="multihead_attention"): 
        '''Applies multihead attention.
    
        Args:
            queries: A 3d tensor with shape of [N, T_q, C_q].
            keys: A 3d tensor with shape of [N, T_k, C_k].
              num_units: A scalar. Attention size.
              causality: Boolean. If true, units that reference the future are masked. 
              num_heads: An int. Number of heads.
              scope: Optional scope for `variable_scope`.
        
        Returns
            A 3d tensor with shape of (N, T_q, C)  
        '''
        with tf.name_scope(scope):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]
        
            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
            # Activation
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
          
            # Dropouts
            outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
               
            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)
              
            # Residual connection
            outputs += queries
              
            # Normalize
            outputs = tf.layers.batch_normalization(outputs, training=self.is_training) # (N, T_q, C)
 
        return outputs

    def feedforward(self, inputs, 
                num_units=[2048, 512],
                scope="multihead_attention"): 
        '''Point-wise feed forward net.
    
        Args:
            inputs: A 3d tensor with shape of [N, T, C].
            num_units: A list of two integers.
            scope: Optional scope for `variable_scope`.
        
        Returns:
            A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.name_scope(scope):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
        
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
        
            # Residual connection
            outputs += inputs
        
            # Normalize
            outputs = tf.layers.batch_normalization(outputs, training=self.is_training)
    
        return outputs

