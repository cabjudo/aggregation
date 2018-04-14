import os
import argparse

import numpy as np
import tensorflow as tf

from convnet.cs231n.data_utils import Datasets
from convnet.cs231n.classifiers.tf_cnn import Networks

from convnet.util.config_reader import get_training
from convnet.util.config_reader import get_logging
from convnet.util.config_reader import get_model
from convnet.util.config_reader import get_dataset


class Evaluator(object):
    def __init__(self, configfile):
        self.model_info = get_model(configfile)
        self.dataset_info = get_dataset(configfile)
        self.training_info = get_training(configfile)
        self.logging_info = get_logging(configfile)

        self.logdir = self.logging_info['params']['logdir']
        self.savedir = self.logging_info['params']['savedir']

        
    def setup(self, seed=0):
        tf.set_random_seed(seed)

        self.setup_data()
        self.setup_network()
        self.setup_logging()


    def setup_data(self):
        with tf.name_scope('data'):
            self.dataset = Datasets[self.dataset_info['options']['data_type']](self.dataset_info)
            
            self.dataset.train_data = self.dataset.train_data.batch(self.training_info['options']['batch_size'])
            
            self.iterator = tf.data.Iterator.from_structure(self.dataset.train_data.output_types, self.dataset.train_data.output_shapes)
            self.img, self.label = self.iterator.get_next()
            
            self.train_init = self.iterator.make_initializer(self.dataset.train_data)
            self.val_init = self.iterator.make_initializer(self.dataset.val_data)
            self.test_init = self.iterator.make_initializer(self.dataset.test_data)

        
    def setup_network(self):
        self.network = Networks[self.model_info['options']['network']](self.img, self.label, self.model_info['params'])
        self.optimize()


    def optimize(self):
        with tf.name_scope('optimize'):
            with tf.name_scope('loss'):
                total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.network.logits)
                self.mean_loss = tf.reduce_mean(total_loss)

            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(self.label, 1), tf.argmax(self.network.logits,1)), tf.float32) )

            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.training_info['options']['lr']).minimize(self.mean_loss, global_step=self.global_step)

            
    def setup_logging(self):
        with tf.name_scope('log'):
            # loss
            self.loss_summary = tf.summary.scalar("loss", self.mean_loss)
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


    def train(self):
        self.num_epochs = self.training_info['options']['num_epochs']
        self.print_every = self.training_info['params']['print_freq']
        self.save_every = self.logging_info['options']['state_freq']
        print(self.training_info['params'])
        self.config = tf.ConfigProto()

        self.saver = tf.train.Saver()
        self.eval(eval_type='TRAIN')

        
    def test(self):
        self.eval(eval_type='TEST')

        
    def eval(self, eval_type='TRAIN'):
        step = 0
        with tf.Session(config=self.config) as sess:
            train_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            val_writer = tf.summary.FileWriter(self.logdir + '/val/', sess.graph)
            
            sess.run(tf.global_variables_initializer())

            # if a checkpoint exists, restore from the latest checkpoint
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.savedir))
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)

            if eval_type == 'TRAIN':
                for _ in range(self.num_epochs):
                   step = self.eval_train(sess, train_writer, step)
                   self.eval_val(sess, val_writer, step)
                    

            if eval_type == 'TEST':
                self.eval_test(sess)

            
    def eval_train(self, sess, writer, step):
        sess.run(self.train_init)
        try:
            while True:
                _, a, l, summary = sess.run([self.optimizer, self.accuracy, self.mean_loss, self.summary_op])
                writer.add_summary(summary, global_step=step)
                if (step % self.print_every) == 0:
                    print('accuracy: ', a, 'loss: ', l, 'step: ', step)
                if (step % self.save_every) == 0:
                    self.saver.save(sess, self.savedir, global_step=self.global_step)
                step += 1
                        
        except tf.errors.OutOfRangeError:
            pass

        return step

        
    def eval_val(self, sess, writer, step):
        sess.run(self.val_init)
        try:
            while True:
                a, loss_summary, accuracy_summary = sess.run([self.accuracy, self.loss_summary, self.accuracy_summary])
                writer.add_summary(accuracy_summary, global_step=step)
                writer.add_summary(loss_summary, global_step=step)
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


            
if __name__ == "__main__":
    configfile = '/home/christine/projects/convnet/config/small_train.ini'

    evaluator = Evaluator(configfile)

    evaluator.setup()

    evaluator.train()
    evaluator.test()
