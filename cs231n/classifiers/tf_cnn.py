import os
import argparse

import tensorflow as tf
import numpy as np

from cs231n.data_utils import get_CIFAR10_data

class Data(object):
    def __init__(self, datadir, num_training=49000, num_validation=1000, num_test=1000 ):
        self.datadir = datadir
        self.num_training = num_training
        self.num_validation = num_validation
        self.num_test = num_test
        
        self.get_data()
        self.build_tf_datasets()
        
    def get_data(self):
        pass

    def build_tf_datasets(self):
        pass
    

class DataCIFAR10(Data):
    def get_data(self):
        self.data = get_CIFAR10_data(cifar10_dir=self.datadir, num_training=self.num_training, num_validation=self.num_validation, num_test=self.num_test)
        
    def build_tf_datasets(self):
        self.train_data = tf.data.Dataset.from_tensor_slices((self.data['X_train'], self.data['y_train']))
        self.train_data = self.train_data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
        # validation data, size: 1000
        self.val_data = tf.data.Dataset.from_tensor_slices((self.data['X_val'], self.data['y_val']))
        self.val_data = self.val_data.batch(self.num_validation)
        self.val_data = self.val_data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
        # test data, size: 1000
        self.test_data = tf.data.Dataset.from_tensor_slices((self.data['X_test'], self.data['y_test']))
        self.test_data = self.test_data.batch(self.num_test)
        self.test_data = self.test_data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
        

Datasets = { 'CIFAR10':DataCIFAR10 }

        
class Baseline(object):
    def __init__(self, logdir, savedir):
        self.feature_maps = {}
        self.logdir = logdir
        self.savedir = savedir
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)#, log_device_placement=True)
        
    def import_data(self, train_init, val_init, test_init, num_epochs):
        self.num_epochs = num_epochs
        with tf.name_scope('data'):
            self.train_init = train_init
            self.val_init = val_init
            self.test_init = test_init

    def logging(self):
        with tf.name_scope('log'):
            # loss
            tf.summary.scalar("loss", self.mean_loss)
            # accuracy
            self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
            
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Summarize all gradients
            grads = tf.gradients(self.mean_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)

            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

            
    def conv_bn_relu_layer(self, input_data, scope, strides=1):
        is_training = True
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=16, kernel_size=3, strides=strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.nn.relu(h, name='relu')

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

            
    def setup(self, train_init, val_init, test_init, num_epochs, img, label):
        self.img = img
        self.label = label
        self.import_data(train_init, val_init, test_init, num_epochs)
        self.predict()
        self.optimize()
        self.logging()

        
    def eval(self, print_every=-1, save_every=-1, seed=0, eval_type='TRAIN'):
        self.print_every = print_every
        self.save_every = save_every
        tf.set_random_seed(seed)
        
        saver = tf.train.Saver()
        step = 0
        with tf.Session(config=self.sess_config) as sess:
            train_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            val_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            
            sess.run(tf.global_variables_initializer())

            # if a checkpoint exists, restore from the latest checkpoint
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.savedir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            if eval_type == 'TRAIN':
                for _ in range(self.num_epochs):
                    self.eval_train(sess, train_writer, step)
                    self.eval_val(sess, val_writer, step)
                    step += 1

            if eval_type == 'TEST':
                self.eval_test(sess)

            
    def eval_train(self, sess, writer, step):
        sess.run(self.train_init)
        try:
            while True:
                _, a, l, summary = sess.run([self.optimizer, self.accuracy, self.mean_loss, self.summary_op])
                writer.add_summary(summary, global_step=step)
                
                if (self.global_step % self.print_every) == 0:
                    print('accuracy: ', a, 'loss: ', l, 'step: ', step)
                    if (self.global_step % self.save_every) == 0:
                        saver.save(sess, self.savedir, global_step=self.global_step)
                        
        except tf.errors.OutOfRangeError:
            pass

        
    def eval_val(self, sess, writer, step):
        sess.run(self.val_init)
        try:
            while True:
                a, summary = sess.run([self.accuracy, self.accuracy_summary])
                writer.add_summary(summary, global_step=step)
                print('validation accuracy: ', a)
                
        except tf.errors.OutOfRangeError:
            pass

            
    def eval_test(self, sess):
        sess.run(self.test_init)
        try:
            while True:
                a = sess.run([self.accuracy])
                print('test accuracy: ', a)
                
        except tf.errors.OutOfRangeError:
            pass

            
            
class ConvModelMaxNorm(Baseline):
    def tf_max_norm(self, x, kernel_size=2):
        with tf.name_scope("max_norm"):
            N, _, _, C = x.get_shape().as_list() # (N, H, W, C)
            
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
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = self.tf_max_norm(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

    
class ConvModelAbs(Baseline):
    def conv_bn_relu_layer(self, input_data, scope, strides=1):
        is_training = True
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=16, kernel_size=3, strides=strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.abs(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a


class ConvModelMaxPool(Baseline):
    def conv_bn_relu_layer(self, input_data, scope, pool_size=2, strides=1):
        is_training = True
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=16, kernel_size=3, strides=strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.layers.max_pooling2d(h, pool_size=pool_size, strides=1, padding='same')

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

    
class ConvModelSelection(Baseline):
    def tf_selection(self, x):
        N, H, W, C = x.get_shape().as_list() # (N, H, W, C)
        
        abs_x = tf.abs(x) # (N, H, W, C)
        
        max_ind = tf.cast(tf.argmax(abs_x, dimension=3), tf.int32) # (N, H, W)
        # print(max_ind.get_shape().as_list())
        max_ind = tf.reshape(max_ind, [-1, H * W])
        # print(max_ind.get_shape().as_list())
        
        H_ind, W_ind = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
        H_ind, W_ind = tf.reshape(H_ind, [-1]), tf.reshape(W_ind, [-1])
        
        out = tf.map_fn( lambda x: tf.gather_nd(x[0], tf.transpose(tf.stack([H_ind, W_ind, x[1]], 0), [1, 0])), (x, max_ind), dtype=tf.float32 )
        # print(out.get_shape().as_list())
        out = tf.reshape(out, (-1, H, W, 1))
        out = x * tf.cast(tf.equal(out, x), tf.float32)
        
        return out

    
    def conv_bn_relu_layer(self, input_data, scope, strides=1):
        is_training = True
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=16, kernel_size=3, strides=strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = self.tf_selection(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a



    
if __name__ == "__main__":
    models = { 'relu':Baseline, 'max_norm':ConvModelMaxNorm, 'abs':ConvModelAbs, 'max':ConvModelMaxPool, 'select':ConvModelSelection }
    
    parser = argparse.ArgumentParser(description='Run CNNs with various nonlinearities')
    parser.add_argument('-network', default='relu', choices=models)
    parser.add_argument('-logdir', default='graphs/')
    parser.add_argument('-savedir', default='checkpoints/')    
    parser.add_argument('-epochs', default=5, type=np.int)
    parser.add_argument('-batch_size', default=64, type=np.int)
    parser.add_argument('-print_every', default=-1, type=np.int)
    parser.add_argument('-save_every', default=-1, type=np.int)
    parser.add_argument('-datadir', default='cs231n/datasets/cifar-10-batches-py', type=str)
    parser.add_argument('-train_size', default=100, type=np.int)
    parser.add_argument('-val_size', default=100, type=np.int)
    parser.add_argument('-test_size', default=100, type=np.int)
    args = parser.parse_args()

    # Invoke the above function to get our data.
    datadir = args.datadir
    num_training = args.train_size
    num_validation = args.val_size
    num_test = args.test_size
    dataset = Datasets['CIFAR10'](datadir, num_training, num_validation, num_test)
    # data = get_CIFAR10_data(cifar10_dir=args.datadir, num_training=num_training, num_validation=num_validation, num_test=num_test)
    # print(data['X_train'].shape, data['y_train'].shape, data['X_val'].shape, data['y_val'].shape, data['X_test'].shape, data['y_test'].shape)
    # preprocessing data
    batch_size = args.batch_size
    dataset.train_data = dataset.train_data.batch(batch_size)
    num_epochs = args.epochs
    # # train data, size: 49000
    # train_data = tf.data.Dataset.from_tensor_slices((data['X_train'], data['y_train']))
    # train_data = train_data.batch(batch_size)
    # train_data = train_data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
    # # validation data, size: 1000
    # val_data = tf.data.Dataset.from_tensor_slices((data['X_val'], data['y_val']))
    # val_data = val_data.batch(args.val_size)
    # val_data = val_data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
    # # test data, size: 1000
    # test_data = tf.data.Dataset.from_tensor_slices((data['X_test'], data['y_test']))
    # test_data = test_data.batch(args.test_size)
    # test_data = test_data.map(lambda x,y: (tf.cast(x, tf.float32), tf.one_hot(y, 10)))
    # construct iterators
    iterator = tf.data.Iterator.from_structure(dataset.train_data.output_types, dataset.train_data.output_shapes)
    img, label = iterator.get_next()
    train_init = iterator.make_initializer(dataset.train_data)
    val_init = iterator.make_initializer(dataset.val_data)
    test_init = iterator.make_initializer(dataset.test_data)


    # Specify the network
    network_name = args.network
    logdir = args.logdir
    savedir = args.savedir
    my_model = models[network_name](logdir, savedir)
    my_model.setup(train_init, val_init, test_init, num_epochs, img, label)
    my_model.eval(print_every=args.print_every, save_every=args.save_every, eval_type='TRAIN')
    my_model.eval(print_every=args.print_every, save_every=args.save_every, eval_type='TEST')
