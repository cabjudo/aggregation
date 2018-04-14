import os
import argparse

import numpy as np
import tensorflow as tf


class TfLogger(object):
    def __init__(self, logging_info): # metric_dict = {'loss': loss, 'accuracy': accuracy}
        self.logging_info = logging_info

    def setup(self):
        self.setup_metric_summaries()
        self.setup_state_summaries()
        self.setup_meta_summaries()

        self.summary_op = tf.summary.merge([self.metrics, self.state, self.meta])


    def associate(self, session):
        self.sess = session

        logdir = self.logging_info['params']['logdir']
        train_writer = tf.summary.FileWriter(logdir, self.sess.graph)
        val_writer = tf.summary.FileWriter(logdir + '/val/', self.sess.graph)

        self.writer = {'TRAIN': train_writer, 'VAL': val_writer }


    def __enter__(self):
        self.saver = tf.train.Saver()
        savedir = self.logging_info['params']['savedir']
        
        # if a checkpoint exists, restore from the latest checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(savedir))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        
    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def step(self, step, save_every, global_step, writer_type='TRAIN'):
        savedir = self.logging_info['params']['savedir']
        
        summary = self.sess.run(self.summary_op)
        self.writer[writer_type].add_summary(summary, global_step=step)

        if ( step % save_every ):
            self.saver.save(self.sess, savedir, global_step=global_step)

        
    def setup_metric_summaries(self):
        self.metrics = []
        for k,v in self.logging_info['metrics'].items():
            self.metrics.append( tf.summary.scalar(k, v) )

    def setup_state_summaries(self):
        self.state = []
        for var in tf.trainable_variables():
            self.state.append( tf.summary.histogram(var.name, var) )

    def setup_meta_summaries(self):
        self.meta = []

        grads = tf.gradients(self.logging_info['metrics']['loss'], tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
        for grad, var in grads:
            self.meta.append( tf.summary.histogram(var.name + '/gradient', grad) )

    


