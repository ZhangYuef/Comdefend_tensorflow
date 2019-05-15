import tensorflow as tf
import numpy as np

class Model_res():
    def __init__(self,n_com=3,n_rec=3,b_com=6,b_rec=6,d_com=2,d_rec=2,com_disable=False,rec_disable=False):
        def make(n,b,d):
            f1 = [2**(b+i) for i in range(n)] + [2**(b+n-1-i) for i in range(n)]
            f2 = [i*(2**d) for i in f1]
            del f2[len(f2)//2]
            f2_last = 32 if f1[0]>32 else 16
            f2.append(f2_last)
            return f1,f2
        
        self.f1_com, self.f2_com = make(n_com,b_com,d_com)
        self.f1_rec, self.f2_rec = make(n_rec,b_rec,d_rec)
        self.n_com = n_com
        self.n_rec = n_rec
        self.com_disable = com_disable
        self.rec_disable = rec_disable

    def res_com(self, X, is_training):
        if self.com_disable:
            inp = X - 0.5
            self.com_conv1 = self.conv2d('com_conv1', inp, 3, 16)
            self.com_conv2 = self.conv2d('com_conv2', self.com_conv1, 16, 32)
            self.com_conv3 = self.conv2d('com_conv3', self.com_conv2, 32, 64)
            self.com_conv4 = self.conv2d('com_conv4', self.com_conv3, 64, 128)
            self.com_conv5 = self.conv2d('com_conv5', self.com_conv4, 128, 256)
            self.com_conv6 = self.conv2d('com_conv6', self.com_conv5, 256, 128)
            self.com_conv7 = self.conv2d('com_conv7', self.com_conv6, 128, 64)
            self.com_conv8 = self.conv2d('com_conv8', self.com_conv7, 64, 32)
            self.com_out = self.conv2d('com_conv9', self.com_conv8, 32, 12, use_elu=False)
        else:
            for i in range(self.n_com*2):
                X = self.res_block(X, [self.f1_com[i],self.f1_com[i],self.f2_com[i]], 'com_'+str(i), is_training)

            self.com_out = tf.contrib.layers.conv2d(X, num_outputs=12, kernel_size=(1, 1),
                                                    stride=(1, 1), padding='VALID', data_format=None,
                                                    activation_fn=None,
                                                    normalizer_fn=None,
                                                    normalizer_params=None,
                                                    weights_initializer=tf.initializers.glorot_uniform,
                                                    weights_regularizer=None)
        
        return self.com_out

    def res_rec(self, X, is_training):
        if self.rec_disable:
            self.rec_conv1 = self.conv2d('rec_conv1', X, 12, 32)
            self.rec_conv2 = self.conv2d('rec_conv2', self.rec_conv1, 32, 64)
            self.rec_conv3 = self.conv2d('rec_conv3', self.rec_conv2, 64, 128)
            self.rec_conv4 = self.conv2d('rec_conv4', self.rec_conv3, 128, 256)
            self.rec_conv5 = self.conv2d('rec_conv5', self.rec_conv4, 256, 128)
            self.rec_conv6 = self.conv2d('rec_conv6', self.rec_conv5, 128, 64)
            self.rec_conv7 = self.conv2d('rec_conv7', self.rec_conv6, 64, 32)
            self.rec_conv8 = self.conv2d('rec_conv8', self.rec_conv7, 32, 16)
            self.rec_conv9 = self.conv2d('rec_conv9', self.rec_conv8, 16, 3, use_elu=False)
            self.rec_out = tf.nn.sigmoid(self.rec_conv9)
        else:
            for i in range(self.n_rec*2):
                X = self.res_block(X, [self.f1_rec[i],self.f1_rec[i],self.f2_rec[i]], 'rec_'+str(i), is_training)

            self.rec_out = tf.contrib.layers.conv2d(X, num_outputs=3, kernel_size=(1, 1),
                                                    stride=(1, 1), padding='VALID', data_format=None,
                                                    activation_fn=tf.nn.sigmoid,
                                                    normalizer_fn=None,
                                                    normalizer_params=None,
                                                    weights_initializer=tf.initializers.glorot_uniform,
                                                    weights_regularizer=None)
        
        return self.rec_out

    def res_block(self, X, filters, name, is_training):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            X1 = self.convolutional_block(X, filters, 'conv', is_training)
            X2 = self.identity_block(X1, filters, 'iden_1', is_training)
            X3 = self.identity_block(X2, filters, 'iden_2', is_training)
            return X3

    def identity_block(self, X, filters, name, is_training, f=3):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            filter_1, filter_2, filter_3 = filters
            X_shortcut = X
            X1 = tf.contrib.layers.conv2d(X, num_outputs=filter_1, kernel_size=(1, 1),
                                         stride=(1, 1), padding='VALID', data_format=None,
                                         activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         weights_initializer=tf.initializers.glorot_uniform,
                                         weights_regularizer=None)
            X2 = tf.contrib.layers.conv2d(X1, num_outputs=filter_2, kernel_size=(f, f),
                                         stride=(1, 1), padding='SAME', data_format=None,
                                         activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         weights_initializer=tf.initializers.glorot_uniform,
                                         weights_regularizer=None)
            X3 = tf.contrib.layers.conv2d(X2, num_outputs=filter_3, kernel_size=(1, 1),
                                         stride=(1, 1), padding='VALID', data_format=None,
                                         activation_fn=None,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         weights_initializer=tf.initializers.glorot_uniform,
                                         weights_regularizer=None)
            X4 = tf.add(X3, X_shortcut)
            X5 = tf.nn.relu(X4)
            return X5

    def convolutional_block(self, X, filters, name, is_training, f=3, s=1):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            filter_1, filter_2, filter_3 = filters
            X_shortcut = X
            X1 = tf.contrib.layers.conv2d(X, num_outputs=filter_1, kernel_size=(1, 1),
                                         stride=(s, s), padding='VALID', data_format=None,
                                         activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         weights_initializer=tf.initializers.glorot_uniform,
                                         weights_regularizer=None)
            X2 = tf.contrib.layers.conv2d(X1, num_outputs=filter_2, kernel_size=(f, f),
                                         stride=(1, 1), padding='SAME', data_format=None,
                                         activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         weights_initializer=tf.initializers.glorot_uniform,
                                         weights_regularizer=None)
            X3 = tf.contrib.layers.conv2d(X2, num_outputs=filter_3, kernel_size=(1, 1),
                                         stride=(1, 1), padding='VALID', data_format=None,
                                         activation_fn=None,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         weights_initializer=tf.initializers.glorot_uniform,
                                         weights_regularizer=None)
            X4 = tf.contrib.layers.conv2d(X_shortcut, num_outputs=filter_3, kernel_size=(1, 1),
                                                  stride=(s, s), padding='VALID', data_format=None,
                                                  activation_fn=None,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  normalizer_params={'is_training': is_training},
                                                  weights_initializer=tf.initializers.glorot_uniform,
                                                  weights_regularizer=None)
            X5 = tf.add(X3, X4)
            X6 = tf.nn.relu(X5)
            return X6

    def conv2d(self, name, x, in_channel, out_channel, use_elu=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            conv_filter = tf.get_variable(name=name+'filter', shape=[3, 3, in_channel, out_channel],
                                          initializer=tf.initializers.truncated_normal(mean=0, stddev=np.sqrt(2/(in_channel*3*3))),
                                          dtype=tf.float32)
            conv = tf.nn.conv2d(x, conv_filter, strides=[1,1,1,1], padding="SAME")
            biases = tf.get_variable(name=name+'bias', shape=[out_channel], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            pre_activation = tf.nn.bias_add(conv, biases)

            if use_elu:
                activation = tf.nn.elu(pre_activation)
                return activation
            else:
                return pre_activation
