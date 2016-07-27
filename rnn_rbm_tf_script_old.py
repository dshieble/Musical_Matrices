import tensorflow as tf
import numpy as np
import pandas as pd
from midi.utils import midiread, midiwrite
import sys
sys.path.append("/Users/danshiebler/Documents")
import helper_functions as hf

from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
from matplotlib import pyplot as plt
import glob
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
from RBM import build_rbm



def get_songs(path='data/music_all/train/*.mid'):
    files = glob.glob(path)[:5]
    f = files[0]
    r=(21, 109)
    dt=0.3
    songs = [midiread(f, r, dt).piano_roll for f in tqdm(files)]
    return songs

songs = get_songs()


def build_rnnrbm( n_visible= 88, n_hidden= 150, n_hidden_recurrent= 100):

    # variables and place holder
    x  = tf.placeholder(tf.float32, [None, n_visible])
    a  = tf.placeholder(tf.float32)
    size_bt = tf.shape(x)[0]

    W  = tf.Variable(tf.random_uniform([n_visible, n_hidden], -0.005, 0.005))
    bh = tf.Variable(tf.zeros([n_hidden], tf.float32))
    bv = tf.Variable(tf.zeros([n_visible], tf.float32))
    Wuh = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden], 0.0001))
    Wuv = tf.Variable(tf.random_normal([n_hidden_recurrent, n_visible], 0.0001))
    Wvu = tf.Variable(tf.random_normal([n_visible, n_hidden_recurrent], 0.0001))
    Wuu = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden_recurrent], 0.0001))
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent],  tf.float32))

    params = W, bh, bv, x, a, Wuh, Wuv, Wvu, Wuu, bu

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    
    def recurrence(xx, u_tm1, _0, _1, count, k):
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))   
        sl   = tf.slice(xx, [count, 0], [1, n_visible])
        u_t  = tf.sigmoid(bu + tf.matmul(sl, Wvu) + tf.matmul(u_tm1, Wuu))

        generate = u_t is None
        if generate:
            x_in = tf.zeros([n_visible, 1],  tf.float32)
            u_t, cost = build_rbm(x_in, W, bv_t, bh_t, k=25)            

        return [xx, u_t, bv_t, bh_t, count+1, k]

    def lessThanNumIter(_0, _1, _2, _3, count, num_iter):
        return count < num_iter

    ct   = tf.constant(1, tf.int32) #counter
    u0   = tf.zeros([1, n_hidden_recurrent], tf.float32)
    bh_t = tf.zeros([1, n_hidden],  tf.float32)#tf.random_uniform([1, n_hidden], -0.005, 0.005)
    bv_t = tf.zeros([1, n_visible],  tf.float32)#tf.random_uniform([1, n_visible], -0.005, 0.005)
    [x, u_t, bv_t, bh_t, _, _] = control_flow_ops.While(lessThanNumIter, recurrence, [x, u0, bv_t, bh_t, ct, size_bt])

    #Build this rbm based on the bias vectors that we already found 
#     W    = tf.Print(W, [W], "W")
#     bv_t    = tf.Print(bv_t, [bv_t], "bv_t")
#     bh_t    = tf.Print(bh_t, [bh_t], "bh_t")   
#     W = tf.Print(W, [bu], "bu")
    sample, cost = build_rbm(x, W, bv_t, bh_t, k=15)
    
    [x, u_t, bv_t, bh_t, _, _] = control_flow_ops.While(lessThanNumIter, lambda u_tm1, *_ : recurrence()
                                                        , [x, u0, bv_t, bh_t, ct, 200])

    
        (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=200)

    
    return x, sample, cost, params, bv_t, bh_t, size_bt

x, sample, cost, params, bv_t, bh_t, size_bt = build_rnnrbm()
W, bh, bv, x, a, Wuh, Wuv, Wvu, Wuu, bu = params

tf.scalar_summary("cost", cost) 
hf.variable_summaries(W, "W")
hf.variable_summaries(bh, "bh")
hf.variable_summaries(bv, "bv")
hf.variable_summaries(Wuh, "Wuh")
hf.variable_summaries(Wuv, "Wuv")
hf.variable_summaries(Wvu, "Wvu")
hf.variable_summaries(Wuu, "Wuu")
hf.variable_summaries(bu, "bu")

# optimizer = tf.train.AdamOptimizer(learning_rate=a).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


num_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/tmp/tf_rbm_logs/train', sess.graph)

    # loop with batch
    for epoch in tqdm(range(num_epochs)):
        for song in songs[:5]:
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                alpha = min(0.05, 100/float(i))
#                 print "iteration: {}, shape: {}".format(i, tr_x.shape)
                summary, _ = sess.run([merged, optimizer], feed_dict={x: tr_x, a: 0.1})
                writer.add_summary(summary, i)


