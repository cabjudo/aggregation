import os
import argparse

import numpy as np
import tensorflow as tf

from cs231n.data_utils import Datasets
from cs231n.classifiers.tf_cnn import Networks


class Evaluator(object):
    def __init__(self, network_type='max', data_type='CIFAR10', logdir=None, savedir=None, config=None):
        self.logdir = logdir
        self.savedir = savedir
        self.config = config

        self.dataset = Datasets[data_type]
        self.network = Networks[network_type]

        
    def setup(self, datadir, num_training=49000, num_validation=1000, num_test=1000, batch_size=64, seed=0):
        self.num_training = num_training
        self.num_validation = num_validation
        self.num_test = num_test
        self.datadir = datadir
        self.setup_data(batch_size)

        tf.set_random_seed(seed)

        self.setup_network()

        self.setup_logging()


    def setup_data(self, batch_size):
        with tf.name_scope('data'):
            self.dataset = self.dataset(self.datadir, num_training=self.num_training, num_validation=self.num_validation, num_test=self.num_test)
            self.dataset.train_data = self.dataset.train_data.batch(batch_size)
            
            self.iterator = tf.data.Iterator.from_structure(self.dataset.train_data.output_types, self.dataset.train_data.output_shapes)
            self.img, self.label = self.iterator.get_next()
            
            self.train_init = self.iterator.make_initializer(self.dataset.train_data)
            self.val_init = self.iterator.make_initializer(self.dataset.val_data)
            self.test_init = self.iterator.make_initializer(self.dataset.test_data)

        
    def setup_network(self):
        self.network = self.network(self.img, self.label)

        
    def setup_logging(self):
        with tf.name_scope('log'):
            # loss
            self.loss_summary = tf.summary.scalar("loss", self.network.mean_loss)
            # accuracy
            self.accuracy_summary = tf.summary.scalar("accuracy", self.network.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Summarize all gradients
            grads = tf.gradients(self.network.mean_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)

            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()


    def train(self, print_every=-1, save_every=-1, num_epochs=1):
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.save_every = save_every

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
                _, a, l, summary = sess.run([self.network.optimizer, self.network.accuracy, self.network.mean_loss, self.summary_op])
                writer.add_summary(summary, global_step=step)
                if (step % self.print_every) == 0:
                    print('accuracy: ', a, 'loss: ', l, 'step: ', step)
                if (step % self.save_every) == 0:
                    self.saver.save(sess, self.savedir, global_step=self.network.global_step)
                step += 1
                        
        except tf.errors.OutOfRangeError:
            pass

        return step

        
    def eval_val(self, sess, writer, step):
        sess.run(self.val_init)
        try:
            while True:
                a, loss_summary, accuracy_summary = sess.run([self.network.accuracy, self.loss_summary, self.accuracy_summary])
                writer.add_summary(accuracy_summary, global_step=step)
                writer.add_summary(loss_summary, global_step=step)
                print('validation accuracy: ', a)
                
        except tf.errors.OutOfRangeError:
            pass

            
    def eval_test(self, sess):
        sess.run(self.test_init)
        try:
            while True:
                a = sess.run([self.network.accuracy])
                print('test accuracy: ', a)
                
        except tf.errors.OutOfRangeError:
            pass


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CNNs with various nonlinearities')
    parser.add_argument('-network', default='relu', choices=Networks)
    parser.add_argument('-logdir', default='graphs/')
    parser.add_argument('-savedir', default='checkpoints/')    
    parser.add_argument('-epochs', default=5, type=np.int)
    parser.add_argument('-batch_size', default=64, type=np.int)
    parser.add_argument('-print_every', default=10, type=np.int)
    parser.add_argument('-save_every', default=100, type=np.int)
    parser.add_argument('-datadir', default='cs231n/datasets/cifar-10-batches-py', type=str)
    parser.add_argument('-train_size', default=1000, type=np.int)
    parser.add_argument('-val_size', default=100, type=np.int)
    parser.add_argument('-test_size', default=100, type=np.int)
    args = parser.parse_args()

    # Invoke the above function to get our data.
    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config=None
    network_name = args.network
    logdir = args.logdir
    savedir = args.savedir
    evaluator = Evaluator(network_type=network_name, data_type='CIFAR10', logdir=logdir, savedir=savedir, config=config)

    datadir = args.datadir
    num_training = args.train_size
    num_validation = args.val_size
    num_test = args.test_size
    batch_size = args.batch_size
    evaluator.setup(datadir, num_training, num_validation, num_test, batch_size)

    num_epochs = args.epochs
    evaluator.train(print_every=args.print_every, save_every=args.save_every, num_epochs=num_epochs)
    evaluator.test()
