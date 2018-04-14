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
        
        print_freq = self.training_info['params']['print_freq']
        self.print_every = steps_per_epoch / print_freq
        save_freq = self.logging_info['options']['state_freq']
        self.save_every = steps_per_epoch / save_freq

        self.config = tf.ConfigProto()

        self.eval(eval_type='TRAIN')

        
    def test(self):
        self.eval(eval_type='TEST')

        
    def eval(self, eval_type='TRAIN'):
        step = 0
        with tf.Session(config=self.config) as sess:
            self.tflogger.associate(sess)
            sess.run(tf.global_variables_initializer())
            with self.tflogger as tfl:
                if eval_type == 'TRAIN':
                    for _ in range(self.num_epochs):
                        step = self.eval_train(sess, step)
                        self.eval_val(sess, step)
                        
                if eval_type == 'TEST':
                    self.eval_test(sess)

            
    def eval_train(self, sess, step):
        sess.run(self.train_init)
        try:
            while True:
                _, a, l = sess.run([self.optimizer, self.accuracy, self.mean_loss])
                if (step % self.print_every) == 0:
                    print('accuracy: ', a, 'loss: ', l, 'step: ', step)

                self.tflogger.step(step, self.save_every, self.global_step, writer_type='TRAIN')
                step += 1
                        
        except tf.errors.OutOfRangeError:
            pass

        return step

        
    def eval_val(self, sess, step):
        sess.run(self.val_init)
        try:
            while True:
                a = sess.run(self.accuracy)
                print('validation accuracy: ', a)
                self.tflogger.step(step, 1, step, writer_type='VAL')
                
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


            
if __name__ == "__main__":
    configfile = '/home/christine/projects/convnet/config/small_train.ini'

    evaluator = Evaluator(configfile)

    evaluator.setup()

    evaluator.train()
    # evaluator.test()
