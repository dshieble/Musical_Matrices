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
from RBM import build_rbm

def get_songs(path='data/music_all/train/*.mid'):
    files = glob.glob(path)[:3]
    f = files[0]
    r=(21, 109)
    dt=0.3
    songs = [midiread(f, r, dt).piano_roll for f in tqdm(files)]
    return songs, dt, r

songs, dt, r = get_songs()


def build_rnnrbm( n_visible= 88, n_hidden= 150, n_hidden_recurrent= 100):

    # variables and place holder
    x  = tf.placeholder(tf.float32, [None, n_visible])
    a  = tf.placeholder(tf.float32)
    size_bt = tf.shape(x)[0]

    W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01))
    bh = tf.Variable(tf.zeros([n_hidden], tf.float32))
    bv = tf.Variable(tf.zeros([n_visible], tf.float32))
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent],  tf.float32))
    Wuh = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden], 0.0001))
    Wuv = tf.Variable(tf.random_normal([n_hidden_recurrent, n_visible], 0.0001))
    Wvu = tf.Variable(tf.random_normal([n_visible, n_hidden_recurrent], 0.0001))
    Wuu = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden_recurrent], 0.0001))

    params = W, bh, bv, x, a, Wuh, Wuv, Wvu, Wuu, bu

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.

    def train_recurrence(count, k, xx, u_tm1, _0, _1):
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))   
        sl   = tf.slice(xx, [count, 0], [1, n_visible])
        u_t  = (tf.tanh(bu + tf.matmul(sl, Wvu) + tf.matmul(u_tm1, Wuu)))
        return count+1, k, xx, u_t, bv_t, bh_t

    def generate_recurrence(count, k, u_tm1, music):
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))   

        x_out, cost  = build_rbm(tf.zeros([1, n_visible],  tf.float32), W, bv_t, bh_t, k=25)
        u_t  = (tf.tanh(bu + tf.matmul(x_out, Wvu) + tf.matmul(u_tm1, Wuu)))
        music = tf.concat(0, [music, x_out])

        return count+1, k, u_t, music

    ct   = tf.constant(1, tf.int32) #counter
    u0   = tf.zeros([1, n_hidden_recurrent], tf.float32)
    bh_t = tf.zeros([1, n_hidden],  tf.float32)#tf.random_uniform([1, n_hidden], -0.005, 0.005)
    bv_t = tf.zeros([1, n_visible],  tf.float32)#tf.random_uniform([1, n_visible], -0.005, 0.005)

    def train(x=x, u0=u0, bv_t=bv_t, bh_t=bh_t, ct=ct, size_bt=size_bt):
        [_, _, x, u_t, bv_t, bh_t] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                                               train_recurrence, 
                                                               [ct, size_bt, x, u0, bv_t, bh_t])

        #Build this rbm based on the bias vectors that we already found 
        sample, cost = build_rbm(x, W, bv_t, bh_t, k=15)
        return x, sample, cost, params, bv_t, bh_t, size_bt

    def generate(u0=u0, bv_t=bv_t, bh_t=bh_t, ct=ct, size_bt=size_bt):
        m = tf.zeros([1, n_visible],  tf.float32)
        [_, _, _, music] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                                         generate_recurrence,
                                                         [ct, tf.constant(200), u0, m])
        return music
    return train, generate


#clip gradients
#don't compute gradients on sample


train, generate = build_rnnrbm()
x, sample, cost, params, bv_t, bh_t, size_bt = train()
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

optimizer = tf.train.AdamOptimizer(learning_rate=a)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
gvs = optimizer.compute_gradients(cost, [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu])
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_var = optimizer.apply_gradients(capped_gvs)





num_epochs = 200
batch_size = 100

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    merged = tf.merge_all_summaries()
    os.system("rm -rf /tmp/tf_rbm_logs/train")
    writer = tf.train.SummaryWriter('/tmp/tf_rbm_logs/train', sess.graph)

    # loop with batch
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                alpha = min(0.05, 100/float(i))
#                 print "iteration: {}, shape: {}".format(i, tr_x.shape)
                summary, _ = sess.run([merged, train_var], feed_dict={x: tr_x, a: 0.1})
                writer.add_summary(summary, i)    
    generated_music = sess.run(generate())
    midiwrite("../music_outputs/composition.midi", generated_music, r, dt)

