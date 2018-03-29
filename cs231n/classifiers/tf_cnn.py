import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt


class LinearModel(object):
    def __init__(self, input_size, output_size):
        with tf.variable_scope('network'):
            self.input_size = input_size
            self.output_size = output_size
            
            self.w = tf.get_variable('weight', [input_size, output_size], initializer=tf.random_normal_initializer())
            self.b = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.0))

        
    def import_data(self, data, labels, num_epochs=1, batch_size=64):
        with tf.name_scope('data'):
            self.data = tf.data.Dataset.from_tensor_slices((data, labels))
            self.data = self.data.batch(batch_size)
            self.data = self.data.repeat(num_epochs)

            self.data = self.data.map(lambda x,y: (tf.cast(tf.reshape(x, [-1, self.input_size]), tf.float32), tf.one_hot(y, 10)))
            
            self.iterator = tf.data.Iterator.from_structure(self.data.output_types)
            self.img, self.label = self.iterator.get_next()
            self.data_init = self.iterator.make_initializer(self.data)

    def logging(self):
        with tf.name_scope('log'):
            tf.summary.scalar("loss", self.mean_loss)
            # tf.summary.scalar("accuracy", accuracy)            
            tf.summary.histogram("weights", tf.reshape(self.w, [1, self.input_size, self.output_size, 1]))
            tf.summary.histogram("biases", self.b)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()
            
    def predict(self):
        with tf.name_scope('predict'):
            logits = tf.matmul(self.img, self.w) + self.b
            total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=logits)
            self.mean_loss = tf.reduce_mean(total_loss)

    def optimize(self, learning_rate=1e-3):
        with tf.name_scope('optimize'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_loss)

            
    def setup(self, data, labels, num_epochs=1, batch_size=64):
        self.import_data(data, labels, num_epochs=num_epochs, batch_size=batch_size)
        self.predict()
        self.optimize()
        
    def train(self, print_every=-1):
        step = 0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('graphs/linear_classifier', sess.graph)
            self.logging()
            
            sess.run(self.data_init)
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    _, l, summary = sess.run([self.optimizer, self.mean_loss, self.summary_op])
                    if (step % print_every)==0:
                        print('loss: ', l, 'step: ', step)
                    writer.add_summary(summary, global_step=step)
                    step += 1
            except tf.errors.OutOfRangeError:
                pass


        
class ConvModel(object):
    def __init__(self):
        self.feature_maps = {}
        
    def import_data(self, data, labels, num_epochs=1, batch_size=64):
        with tf.name_scope('data'):
            self.data = tf.data.Dataset.from_tensor_slices((data, labels))
            self.data = self.data.batch(batch_size)
            self.data = self.data.repeat(num_epochs)

            self.data = self.data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
            
            self.iterator = tf.data.Iterator.from_structure(self.data.output_types)
            self.img, self.label = self.iterator.get_next()
            self.data_init = self.iterator.make_initializer(self.data)

    def logging(self):
        with tf.name_scope('log'):
            # loss
            tf.summary.scalar("loss", self.mean_loss)
            # accuracy
            tf.summary.scalar("accuracy", self.accuracy)
            
            # weights
            tf.summary.histogram("weight_conv1", tf.trainable_variables('network/layer1/conv/kernel')[0])
            tf.summary.histogram("weight_conv2", tf.trainable_variables('network/layer2/conv/kernel')[0])
            tf.summary.histogram("weight_conv3", tf.trainable_variables('network/layer3/conv/kernel')[0])
            tf.summary.histogram("weight_conv4", tf.trainable_variables('network/layer4/conv/kernel')[0])
            tf.summary.histogram("weight_conv5", tf.trainable_variables('network/layer5/conv/kernel')[0])
            tf.summary.histogram("weight_conv6", tf.trainable_variables('network/layer6/conv/kernel')[0])

            # bias
            tf.summary.histogram("bias_conv1", tf.trainable_variables('network/layer1/conv/bias')[0])
            tf.summary.histogram("bias_conv2", tf.trainable_variables('network/layer2/conv/bias')[0])
            tf.summary.histogram("bias_conv3", tf.trainable_variables('network/layer3/conv/bias')[0])
            tf.summary.histogram("bias_conv4", tf.trainable_variables('network/layer4/conv/bias')[0])
            tf.summary.histogram("bias_conv5", tf.trainable_variables('network/layer5/conv/bias')[0])
            tf.summary.histogram("bias_conv6", tf.trainable_variables('network/layer6/conv/bias')[0])

            # feature maps
            tf.summary.histogram('feature_map1', self.feature_maps['layer1_h'])
            tf.summary.histogram('feature_map2', self.feature_maps['layer2_h'])
            tf.summary.histogram('feature_map3', self.feature_maps['layer3_h'])
            tf.summary.histogram('feature_map4', self.feature_maps['layer4_h'])
            tf.summary.histogram('feature_map5', self.feature_maps['layer5_h'])
            tf.summary.histogram('feature_map6', self.feature_maps['layer6_h'])
            
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def conv_bn_relu_layer(self, input_data, scope, strides=1):
        is_training = True
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=16, kernel_size=3, strides=strides, padding='valid', name='conv') # 30x30x64
            nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.nn.relu(nh, name='relu')

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

            
    def predict(self):
        with tf.variable_scope('network'):
            a1 = self.conv_bn_relu_layer(tf.reshape(self.img, [-1,32,32,3]), 'layer1')
            a2 = self.conv_bn_relu_layer(a1, 'layer2')
            a3 = self.conv_bn_relu_layer(a2, 'layer3', strides=2)
            a4 = self.conv_bn_relu_layer(a3, 'layer4')
            a5 = self.conv_bn_relu_layer(a4, 'layer5')
            a6 = self.conv_bn_relu_layer(a5, 'layer6', strides=2)

            self.activations = [a1, a2, a3, a4, a5, a6]

            with tf.variable_scope('layer7'):
                h7 = tf.layers.conv2d(a6, filters=10, kernel_size=4, padding='valid', name='conv') # 1x1x10
            
        with tf.name_scope('loss'):
            logits = tf.reshape(h7,[-1,10])
            total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=logits)
            self.mean_loss = tf.reduce_mean(total_loss)

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(self.label, 1), tf.argmax(logits,1)), tf.float32) )

    def optimize(self, learning_rate=1e-3):
        with tf.name_scope('optimize'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_loss)

            
    def setup(self, data, labels, num_epochs=1, batch_size=64):
        self.import_data(data, labels, num_epochs=num_epochs, batch_size=batch_size)
        self.predict()
        self.optimize()
        
    def train(self, print_every=100):
        step = 0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('graphs/baseline', sess.graph)
            self.logging()
            
            sess.run(self.data_init)
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    _, l, summary = sess.run([self.optimizer, self.mean_loss, self.summary_op])
                    if (step % print_every)==0:
                        print('loss: ', l, 'step: ', step)
                    writer.add_summary(summary, global_step=step)
                    step += 1
            except tf.errors.OutOfRangeError:
                pass
        

class ConvModelMaxNorm(ConvModel):
    # def __init__(self):
    #     self.feature_maps = {}
        
    # def import_data(self, data, labels, num_epochs=1, batch_size=64):
    #     with tf.name_scope('data'):
    #         self.data = tf.data.Dataset.from_tensor_slices((data, labels))
    #         self.data = self.data.batch(batch_size)
    #         self.data = self.data.repeat(num_epochs)

    #         self.data = self.data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
            
    #         self.iterator = tf.data.Iterator.from_structure(self.data.output_types)
    #         self.img, self.label = self.iterator.get_next()
    #         self.data_init = self.iterator.make_initializer(self.data)

    # def logging(self):
    #     with tf.name_scope('log'):
    #         tf.summary.scalar("loss", self.mean_loss)
            
    #         # weights
    #         tf.summary.histogram("weight_conv1", tf.trainable_variables('network/layer1/conv/kernel')[0])
    #         tf.summary.histogram("weight_conv2", tf.trainable_variables('network/layer2/conv/kernel')[0])
    #         tf.summary.histogram("weight_conv3", tf.trainable_variables('network/layer3/conv/kernel')[0])
    #         tf.summary.histogram("weight_conv4", tf.trainable_variables('network/layer4/conv/kernel')[0])
    #         tf.summary.histogram("weight_conv5", tf.trainable_variables('network/layer5/conv/kernel')[0])
    #         tf.summary.histogram("weight_conv6", tf.trainable_variables('network/layer6/conv/kernel')[0])

    #         # bias
    #         tf.summary.histogram("bias_conv1", tf.trainable_variables('network/layer1/conv/bias')[0])
    #         tf.summary.histogram("bias_conv2", tf.trainable_variables('network/layer2/conv/bias')[0])
    #         tf.summary.histogram("bias_conv3", tf.trainable_variables('network/layer3/conv/bias')[0])
    #         tf.summary.histogram("bias_conv4", tf.trainable_variables('network/layer4/conv/bias')[0])
    #         tf.summary.histogram("bias_conv5", tf.trainable_variables('network/layer5/conv/bias')[0])
    #         tf.summary.histogram("bias_conv6", tf.trainable_variables('network/layer6/conv/bias')[0])

    #         # feature maps
    #         tf.summary.histogram('feature_map1', self.feature_maps['layer1_h'])
    #         tf.summary.histogram('feature_map2', self.feature_maps['layer2_h'])
    #         tf.summary.histogram('feature_map3', self.feature_maps['layer3_h'])
    #         tf.summary.histogram('feature_map4', self.feature_maps['layer4_h'])
    #         tf.summary.histogram('feature_map5', self.feature_maps['layer5_h'])
    #         tf.summary.histogram('feature_map6', self.feature_maps['layer6_h'])
            
    #         # because you have several summaries, we should merge them all
    #         # into one op to make it easier to manage
    #         self.summary_op = tf.summary.merge_all()

    
    def tf_max_norm(self, x, kernel_size=2):
        N, _, _, C = x.get_shape().as_list() # (N, C, H, W)
        
        ksizes = [1, kernel_size, kernel_size, 1]
        strides = [1, 1, 1, 1]
        rates = [1, 1, 1, 1]
        
        patches = tf.extract_image_patches(x, ksizes, strides, rates, padding="SAME")
        _, nh, nw, D = patches.get_shape().as_list() # (N, num_patches_v, num_patches_h, D), D = C * kernel_size**2
        patches = tf.reshape(patches, (-1, nh, nw, kernel_size**2, C)) # (N, nh, nw, kernel_size**2, C)

        abs_patches = tf.abs(patches) # (N, nh, nw, kernel_size**2, C)
        
        abs_max = tf.cast(tf.argmax(abs_patches, dimension=3), tf.int32) # (N, nh, nw, C)
        abs_max = tf.reshape(abs_max, [-1, nh * nw * 1 * C])
        
        nh_ind, nw_ind, C_ind = tf.meshgrid(tf.range(nh), tf.range(nw), tf.range(C), indexing='ij')
        nh_ind, nw_ind, C_ind = tf.reshape(nh_ind, [-1]), tf.reshape(nw_ind, [-1]), tf.reshape(C_ind, [-1])
        
        out = tf.map_fn( lambda x: tf.gather_nd(x[0], tf.transpose(tf.stack([nh_ind, nw_ind, x[1], C_ind], 0), [1, 0])), (patches, abs_max), dtype=tf.float32 )
        out = tf.reshape(out, (-1, nh, nw, C))
        
        return out

    
    def conv_bn_relu_layer(self, input_data, scope, strides=1):
        is_training = True
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=16, kernel_size=3, strides=strides, padding='valid', name='conv') # 30x30x64
            nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = self.tf_max_norm(nh)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

            
    # def predict(self):
    #     with tf.variable_scope('network'):
    #         a1 = self.conv_bn_relu_layer(tf.reshape(self.img, [-1,32,32,3]), 'layer1')
    #         a2 = self.conv_bn_relu_layer(a1, 'layer2')
    #         a3 = self.conv_bn_relu_layer(a2, 'layer3', strides=2)
    #         a4 = self.conv_bn_relu_layer(a3, 'layer4')
    #         a5 = self.conv_bn_relu_layer(a4, 'layer5')
    #         a6 = self.conv_bn_relu_layer(a5, 'layer6', strides=2)

    #         self.activations = [a1, a2, a3, a4, a5, a6]

    #         with tf.variable_scope('layer7'):
    #             h7 = tf.layers.conv2d(a6, filters=10, kernel_size=4, padding='valid', name='conv') # 1x1x10
            
    #     with tf.name_scope('loss'):
    #         logits = tf.reshape(h7,[-1,10])
    #         total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=logits)
    #         self.mean_loss = tf.reduce_mean(total_loss)

    # def optimize(self, learning_rate=1e-3):
    #     with tf.name_scope('optimize'):
    #         self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_loss)

            
    # def setup(self, data, labels, num_epochs=1, batch_size=64):
    #     self.import_data(data, labels, num_epochs=num_epochs, batch_size=batch_size)
    #     self.predict()
    #     self.optimize()
        
    def train(self, print_every=100):
        step = 0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('graphs/max_norm', sess.graph)
            self.logging()
            
            sess.run(self.data_init)
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    _, l, summary = sess.run([self.optimizer, self.mean_loss, self.summary_op])
                    if (step % print_every)==0:
                        print('loss: ', l, 'step: ', step)
                    writer.add_summary(summary, global_step=step)
                    step += 1
            except tf.errors.OutOfRangeError:
                pass
