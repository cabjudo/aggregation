import os
import argparse

import numpy as np
import tensorflow as tf

from convnet.util.data_utils import Datasets
from convnet.classifiers.tf_cnn import Networks
from convnet.util.tf_logging import TfLogger

from convnet.util.config_reader import get_training
from convnet.util.config_reader import get_logging
from convnet.util.config_reader import get_model
from convnet.util.config_reader import get_dataset

from convnet.util.graceful_exit import GracefulExit


class Evaluator(object):
    def __init__(self, configfile):
        self.model_info = get_model(configfile)
        self.dataset_info = get_dataset(configfile)
        self.training_info = get_training(configfile)
        self.logging_info = get_logging(configfile)


    def setup(self, seed=0):
        tf.set_random_seed(seed)

        self.setup_data()
        self.setup_network()
        self.setup_logging()


    def setup_data(self):
        data_type = self.dataset_info['options']['data_type']
        batch_size = self.training_info['options']['batch_size']
        
        with tf.name_scope('data'):
            self.dataset = Datasets[data_type](self.dataset_info)
            self.dataset.train_data = self.dataset.train_data.batch(batch_size)
            
            self.iterator = tf.data.Iterator.from_structure(self.dataset.train_data.output_types, self.dataset.train_data.output_shapes)
            self.img, self.label = self.iterator.get_next()
            
            self.train_init = self.iterator.make_initializer(self.dataset.train_data)
            self.val_init = self.iterator.make_initializer(self.dataset.val_data)
            self.test_init = self.iterator.make_initializer(self.dataset.test_data)

        
    def setup_network(self):
        network = self.model_info['options']['network']
        self.network = Networks[network](self.img, self.label, self.model_info['params'])
        
        self.optimize()


    def setup_logging(self):
        metrics = {'loss': self.mean_loss, 'accuracy': self.accuracy }
        self.logging_info['metrics'] = metrics

        self.tflogger = TfLogger(self.logging_info)

        with tf.name_scope('log'):
            self.tflogger.setup()
            

    def optimize(self):
        with tf.name_scope('optimize'):
            with tf.name_scope('loss'):
                total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.network.logits)
                self.mean_loss = tf.reduce_mean(total_loss)

            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(self.label, 1), tf.argmax(self.network.logits,1)), tf.float32) )

            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.training_info['options']['lr']).minimize(self.mean_loss, global_step=self.global_step)


    def train(self):
        self.num_epochs = self.training_info['options']['num_epochs']
        train_size = self.dataset_info['params']['train_size']
        batch_size = self.training_info['options']['batch_size']
        steps_per_epoch = np.ceil( train_size / batch_size )

        # convert from print_freq to print_every
        print_freq = self.training_info['params']['print_freq']
        self.print_every = steps_per_epoch / print_freq

        # convert from save_freq to save_every
        save_freq = self.logging_info['options']['state_freq']
        self.save_every = steps_per_epoch / save_freq

        self.config = tf.ConfigProto()

        self.evaluate(eval_type='TRAIN')

        
    def test(self):
        self.evaluate(eval_type='TEST')

        
    def evaluate(self, eval_type='TRAIN'):
        self.step = 0
        self.sess = tf.Session(config=self.config)
        
        # associate the logger with this session
        self.tflogger.associate(self.sess)
        
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # with logger as context manager
        with self.tflogger as tfl:
            if eval_type == 'TRAIN':
                for _ in range(self.num_epochs):
                    # train for num_epochs
                    self.step = self.eval_train()
                    # look at results on validation after each epoch
                    self.eval_val()
                    
            if eval_type == 'TEST':
                self.eval_test()
        # close the session
        self.sess.close()

            
    def eval_train(self):
        # initialize iterator
        self.sess.run(self.train_init)
        try:
            while True:
                # train (optim), and report accuracy and loss for user
                _, a, l = self.sess.run([self.optimizer, self.accuracy, self.mean_loss])
                if (self.step % self.print_every) == 0:
                    print('accuracy: ', a, 'loss: ', l, 'step: ', self.step)
                # notify logger, it will save if necessary
                self.tflogger.step(self.step, self.save_every, self.global_step, writer_type='TRAIN')
                self.step += 1
                        
        except tf.errors.OutOfRangeError:
            # print('eval_train: caught outofrangeerror')
            pass

        return self.step

        
    def eval_val(self):
        self.sess.run(self.val_init)
        try:
            while True:
                a = self.sess.run(self.accuracy)
                print('validation accuracy: ', a)
                self.tflogger.step(self.step, 1, self.global_step, writer_type='VAL')
                
        except tf.errors.OutOfRangeError:
            # print('eval_val: caught outofrangeerror')
            pass

            
    def eval_test(self):
        self.sess.run(self.test_init)
        try:
            while True:
                a = self.sess.run([self.accuracy])
                print('test accuracy: ', a)
                
        except tf.errors.OutOfRangeError:
            # print('eval_test: caught outofrangeerror')
            pass


    def clean_up(self):
        try:
            # print('in trainers cleanup routine')
            pass
        except tf.errors.OutOfRangeError:
            # print('clean_up: caught OutOfRangeError')
            pass
        finally:
            # print('in trainers cleanup routine')
            self.tflogger.step(self.step, 1, self.global_step, writer_type='TRAIN')
            self.tflogger.sess.close()

        
            
if __name__ == "__main__":
    from convnet.util.graceful_exit import GracefulExit
    
    configfile = '/home/christine/projects/convnet/config/small_train.ini'
    time_limit = 60*60

    evaluator = Evaluator(configfile)
    evaluator.setup()

    with GracefulExit(time_limit, evaluator) as ge:
        ge.evaluate()

    evaluator.test()
