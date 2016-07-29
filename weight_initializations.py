#Initialize the weight and bias vectors for the RNN and RBM
import tensorflow as tf
import numpy as np
import pandas as pd
from midi.utils import midiread, midiwrite
import sys
sys.path.append("/Users/danshiebler/Documents")
import helper_functions as hf
import os
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
from matplotlib import pyplot as plt
import glob
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import RBM
import rnn_rbm

#Initialize the RBM Variables
songs, dt, r = rnn_rbm.get_songs('data/Nottingham/train/*.mid', 1000)

n_visible= 88
n_hidden= 150
n_hidden_recurrent= 100
batch_size = 100

# variables and place holder
x  = tf.placeholder(tf.float32, [None, n_visible], name="x")
W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")

bh_t = tf.zeros([1, n_hidden],  tf.float32, name="bh_t")
bv_t = tf.zeros([1, n_visible],  tf.float32, name="bv_t")

size_bt = tf.shape(x)[0]

saver = tf.train.Saver()
xk1, cost = RBM.build_rbm(x, W, bv_t, bh_t, 1)
rbm_train_var = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#Initialize the RNN Variables


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    # loop with batch
    for song in tqdm(songs):
        for i in range(1, len(song), batch_size):
            tr_x = song[i:i + batch_size]
            sess.run(rbm_train_var, feed_dict={x: tr_x})
    save_path = saver.save(sess, "initializations/rbm.ckpt")


