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
    songs = [midiread(f, r, dt).piano_roll for f in tqdm(files)]
    return songs, dt, r



def build_rnnrbm( n_visible= 88, n_hidden= 150, n_hidden_recurrent= 100):

    # variables and place holder
    x  = tf.placeholder(tf.float32, [None, n_visible])
    a  = tf.placeholder(tf.float32)
    size_bt = tf.shape(x)[0]

    W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
    Wuh = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden], 0.0001), name="Wuh")
    Wuv = tf.Variable(tf.random_normal([n_hidden_recurrent, n_visible], 0.0001), name="Wuv")
    Wvu = tf.Variable(tf.random_normal([n_visible, n_hidden_recurrent], 0.0001), name="Wvu")
    Wuu = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden_recurrent], 0.0001), name="Wuu")
    bh = tf.Variable(tf.zeros([n_hidden], tf.float32), name="bh")
    bv = tf.Variable(tf.zeros([n_visible], tf.float32), name="bv")
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent],  tf.float32), name="bu")
    
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

        x_out  = RBM.gibbs_sample(tf.zeros([1, n_visible],  tf.float32), W, bv_t, bh_t, k=25)
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
        #TODO: Replace with simple sampling?
        sample, cost = RBM.build_rbm(x, W, bv_t, bh_t, k=15)
        return x, sample, cost, params, bv_t, bh_t, size_bt

    def generate(u0=u0, bv_t=bv_t, bh_t=bh_t, ct=ct, size_bt=size_bt):
        m = tf.zeros([1, n_visible],  tf.float32)
        [_, _, _, music] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                                         generate_recurrence,
                                                         [ct, tf.constant(200), u0, m])
        return music
    return train, generate

