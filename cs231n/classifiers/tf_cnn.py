import os
import argparse

import tensorflow as tf
import numpy as np

from cs231n.data_utils import get_CIFAR10_data

        
class Baseline(object):
    def __init__(self, logdir, savedir):
        self.feature_maps = {}
        self.logdir = logdir
        self.savedir = savedir
        
    def import_data(self, data, labels, num_epochs, batch_size):
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        
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
            
            # # weights
            # tf.summary.histogram("weight_conv1", tf.trainable_variables('network/layer1/conv/kernel')[0])
            # tf.summary.histogram("weight_conv2", tf.trainable_variables('network/layer2/conv/kernel')[0])
            # tf.summary.histogram("weight_conv3", tf.trainable_variables('network/layer3/conv/kernel')[0])
            # tf.summary.histogram("weight_conv4", tf.trainable_variables('network/layer4/conv/kernel')[0])
            # tf.summary.histogram("weight_conv5", tf.trainable_variables('network/layer5/conv/kernel')[0])
            # tf.summary.histogram("weight_conv6", tf.trainable_variables('network/layer6/conv/kernel')[0])

            # # bias
            # tf.summary.histogram("bias_conv1", tf.trainable_variables('network/layer1/conv/bias')[0])
            # tf.summary.histogram("bias_conv2", tf.trainable_variables('network/layer2/conv/bias')[0])
            # tf.summary.histogram("bias_conv3", tf.trainable_variables('network/layer3/conv/bias')[0])
            # tf.summary.histogram("bias_conv4", tf.trainable_variables('network/layer4/conv/bias')[0])
            # tf.summary.histogram("bias_conv5", tf.trainable_variables('network/layer5/conv/bias')[0])
            # tf.summary.histogram("bias_conv6", tf.trainable_variables('network/layer6/conv/bias')[0])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Summarize all gradients
            grads = tf.gradients(self.mean_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)

            # # feature maps
            # tf.summary.histogram('feature_map1', self.feature_maps['layer1_h'])
            # tf.summary.histogram('feature_map2', self.feature_maps['layer2_h'])
            # tf.summary.histogram('feature_map3', self.feature_maps['layer3_h'])
            # tf.summary.histogram('feature_map4', self.feature_maps['layer4_h'])
            # tf.summary.histogram('feature_map5', self.feature_maps['layer5_h'])
            # tf.summary.histogram('feature_map6', self.feature_maps['layer6_h'])
            
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
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_loss, global_step=self.global_step)

            
    def setup(self, data, labels, num_epochs=1, batch_size=64):
        self.import_data(data, labels, num_epochs=num_epochs, batch_size=batch_size)
        self.predict()
        self.optimize()
        
    def train(self, print_every=-1, save_every=-1, seed=0):
        print('epochs: ', self.num_epochs, 'batch size: ', self.batch_size)
        tf.set_random_seed(seed)
        saver = tf.train.Saver()
        
        step = 0
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.logdir, sess.graph)
            self.logging()
            
            sess.run(self.data_init)
            sess.run(tf.global_variables_initializer())

            # if a checkpoint exists, restore from the latest checkpoint
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.savedir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            try:
                while True:
                    _, a, l, summary = sess.run([self.optimizer, self.accuracy, self.mean_loss, self.summary_op])
                    writer.add_summary(summary, global_step=step)
                    step += 1

                    if (step % print_every) == 0:
                        print('accuracy: ', a, 'loss: ', l, 'step: ', step)
                    if (step % save_every) == 0:
                        saver.save(sess, self.savedir, global_step=self.global_step)
                        
            except tf.errors.OutOfRangeError:
                pass
        

class ConvModelMaxNorm(Baseline):
    def tf_max_norm(self, x, kernel_size=2):
        with tf.name_scope("max_norm"):
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

    
class ConvModelAbs(Baseline):
    def conv_bn_relu_layer(self, input_data, scope, strides=1):
        is_training = True
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=16, kernel_size=3, strides=strides, padding='valid', name='conv') # 30x30x64
            nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.abs(nh)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

    
            
if __name__ == "__main__":
    models = {'relu':Baseline, 'max_norm':ConvModelMaxNorm, 'abs':ConvModelAbs}
    
    parser = argparse.ArgumentParser(description='Run CNNs with various nonlinearities')
    parser.add_argument('-network', default='relu', choices=models)
    # parser.add_argument('-logdir', default='graphs/relu/')
    # parser.add_argument('-savedir', default='checkpoints/relu/')    
    parser.add_argument('-epochs', default=5, type=np.int)
    parser.add_argument('-batch_size', default=64, type=np.int)
    parser.add_argument('-print_every', default=-1, type=np.int)
    parser.add_argument('-save_every', default=-1, type=np.int)
    args = parser.parse_args()

    # Invoke the above function to get our data.
    # X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    data = get_CIFAR10_data()
    # print('Train data shape: ', data['X_train'].shape)
    # print('Train labels shape: ', data['y_train'].shape)
    # print('Validation data shape: ', data['X_val'].shape)
    # print('Validation labels shape: ', data['y_val'].shape)
    # print('Test data shape: ', data['X_test'].shape)
    # print('Test labels shape: ', data['y_test'].shape)

    network_name = args.network
    logdir = 'graphs/' + network_name
    savedir = 'checkpoints/' + network_name
    
    my_model = models[network_name](logdir, savedir)
    my_model.setup(data['X_train'], data['y_train'], num_epochs=args.epochs, batch_size=args.batch_size)
    my_model.train(print_every=args.print_every, save_every=args.save_every)
