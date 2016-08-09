#Initialize the weight and bias vectors for the RNN and RBM
import tensorflow as tf
import numpy as np
import pandas as pd
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

#Load the Songs
# songs = rnn_rbm.get_songs('data/Nottingham/all_C/*.mid', 1000)
songs = rnn_rbm.get_songs_msgpack('data/song_msgpacks/biaxial_all_C', 1000)

batch_size = 100


x_rbm  = tf.placeholder(tf.float32, [None, rnn_rbm.n_visible], name="x_rbm")
x_rnn  = tf.placeholder(tf.float32, [None, rnn_rbm.n_visible], name="x")
size_bt = tf.shape(x_rnn)[0]

#RBM  and RBM Variables
W   = tf.Variable(tf.random_normal([rnn_rbm.n_visible, rnn_rbm.n_hidden], 0.01), name="W")
Wuh = tf.Variable(tf.random_normal([rnn_rbm.n_hidden_recurrent, rnn_rbm.n_hidden], 0.0001), name="Wuh")
Wuv = tf.Variable(tf.random_normal([rnn_rbm.n_hidden_recurrent, rnn_rbm.n_visible], 0.0001), name="Wuv")
Wvu = tf.Variable(tf.random_normal([rnn_rbm.n_visible, rnn_rbm.n_hidden_recurrent], 0.0001), name="Wvu")
Wuu = tf.Variable(tf.random_normal([rnn_rbm.n_hidden_recurrent, rnn_rbm.n_hidden_recurrent], 0.0001), name="Wuu")
bh  = tf.Variable(tf.zeros([1, rnn_rbm.n_hidden], tf.float32), name="bh")
bv  = tf.Variable(tf.zeros([1, rnn_rbm.n_visible], tf.float32), name="bv")
bu  = tf.Variable(tf.zeros([1, rnn_rbm.n_hidden_recurrent],  tf.float32), name="bu")
u0  = tf.Variable(tf.zeros([1, rnn_rbm.n_hidden_recurrent], tf.float32), name="u0")
BH_t = tf.Variable(tf.ones([1, rnn_rbm.n_hidden],  tf.float32), name="BH_t")
BV_t = tf.Variable(tf.ones([1, rnn_rbm.n_visible],  tf.float32), name="BV_t")




size_bt = tf.shape(x_rbm)[0]

#Build the RBM optimization
saver = tf.train.Saver()
# xk1, cost = RBM.get_free_energy_cost(x_rbm, W, bv, bh, 1)
# updt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
updt = RBM.get_cd_update(x_rbm, W, bv, bh, 1, 0.01)

#Run the session
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    # loop with batch
    for epoch in tqdm(range(10)):
        for song in songs:
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x_rbm: tr_x})
    save_path = saver.save(sess, "initializations/rbm.ckpt")

 
