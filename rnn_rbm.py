import sys
import tensorflow as tf
import numpy as np
import pandas as pd
sys.path.append("midi")
from utils import midiread, midiwrite
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

def get_songs(path, num_songs):
    
    files = glob.glob(path)[:num_songs]
    f = files[0]
    r=(21, 109)
    dt=0.3
    songs = []
    for f in tqdm(files):
        try:
            songs.append(midiread(f, r, dt).piano_roll)
        except Exception as e:
            print f, e            
    return songs, dt, r



def build_rnnrbm( n_visible= 88, n_hidden= 150, n_hidden_recurrent= 100):

    # variables and place holder
    x  = tf.placeholder(tf.float32, [None, n_visible])
    a  = tf.placeholder(tf.float32)
    
    size_bt = tf.shape(x)[0]

    W   = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
    Wuh = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden], 0.0001), name="Wuh")
    Wuv = tf.Variable(tf.random_normal([n_hidden_recurrent, n_visible], 0.0001), name="Wuv")
    Wvu = tf.Variable(tf.random_normal([n_visible, n_hidden_recurrent], 0.0001), name="Wvu")
    Wuu = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden_recurrent], 0.0001), name="Wuu")
    bh  = tf.Variable(tf.zeros([1, n_hidden], tf.float32), name="bh")
    bv  = tf.Variable(tf.zeros([1, n_visible], tf.float32), name="bv")
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent],  tf.float32), name="bu")
    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent], tf.float32), name="u0")
    
    params = W, bh, bv, x, a, Wuh, Wuv, Wvu, Wuu, bu, u0

    BH_t = tf.Variable(tf.zeros([1, n_hidden], tf.float32), name="BH_t")
    BV_t = tf.Variable(tf.zeros([1, n_visible], tf.float32), name="BV_t")
    #At graph building time 
    tf.assign(BH_t, tf.tile(BH_t, [size_bt, 1]))
    tf.assign(BV_t, tf.tile(BV_t, [size_bt, 1]))

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def rnn_recurrence(u_tm1, sl):
        sl  =  tf.reshape(sl, [1, n_visible])
        u_t = (tf.tanh(bu + tf.matmul(sl, Wvu) + tf.matmul(u_tm1, Wuu)))
        return u_t
    
    def visible_bias_recurrence(bv_t, u_tm1):
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        return bv_t    

    def hidden_bias_recurrence(bh_t, u_tm1):
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))
        return bh_t    

    def train(x=x, size_bt=size_bt, BV_t=BV_t, BH_t=BH_t):
        bv_init = tf.zeros([1, n_visible], tf.float32)
        bh_init = tf.zeros([1, n_hidden], tf.float32)
        u_t  = tf.scan(rnn_recurrence, x, initializer=u0)
        BV_t = tf.reshape(tf.scan(visible_bias_recurrence, u_t, bv_init), [size_bt, n_visible])
        BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, u_t, bh_init), [size_bt, n_hidden])
        sample, cost = RBM.build_rbm(x, W, BV_t, BH_t, k=15)
        return x, sample, cost, params, size_bt            
        
        
#         [_, _, u_t] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
#                                                                train_recurrence, 
#                                                                [ct, size_bt, u0])

        #Build this rbm based on the bias vectors that we already found 
#         BV_t = tf.Print(BV_t, [BV_t], "BV_t", summarize=10)
#         BH_t = tf.Print(BH_t, [BH_t], "BH_t", summarize=10)
#         BV_t = tf.get_variable("BV_t",[size_bt, n_visible],dtype=tf.float32)
#         BH_t = tf.get_variable("BH_t",[size_bt, n_hidden] ,dtype=tf.float32)

#         with tf.control_dependencies([u_t]):

#             sample, cost = RBM.build_rbm(x, W, BV_t, BH_t, k=15)
#             return x, sample, cost, params, size_bt

    def generate_recurrence(count, k, u_tm1, music):
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))   

        x_out  = RBM.gibbs_sample(tf.zeros([1, n_visible],  tf.float32), W, bv_t, bh_t, k=25)
        u_t  = (tf.tanh(bu + tf.matmul(x_out, Wvu) + tf.matmul(u_tm1, Wuu)))
        music = tf.concat(0, [music, x_out])

        return count+1, k, u_t, music

    def generate(u0=u0, size_bt=size_bt):
        m = tf.zeros([1, n_visible],  tf.float32)
        ct   = tf.constant(1, tf.int32) #counter
        [_, _, _, music] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                                         generate_recurrence,
                                                         [ct, tf.constant(200), u0, m])
        return music
    return train, generate

