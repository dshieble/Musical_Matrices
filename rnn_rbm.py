import sys
import tensorflow as tf
import numpy as np
import pandas as pd
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
import midi_to_statematrix
import msgpack

n_visible= 156
n_hidden= 250
n_hidden_recurrent= 100

def write_song(path, song):
    midi_to_statematrix.noteStateMatrixToMidi(song, name=path)

def get_song(path):
    statematrix  = np.array(midi_to_statematrix.midiToNoteStateMatrix(path))
    return statematrix

def get_songs(path, num_songs):
        
    files = glob.glob('{}/*.mid'.format(path))[:num_songs]
    songs = []
    for f in tqdm(files):
        try:
            song = get_song(f)
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            print f, e            
    return songs

def get_songs_msgpack(path, num_songs):
    return msgpack.load(open(path, "rb"))[:num_songs]


def build_rnnrbm():

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
        sample, cost, monitor = RBM.get_free_energy_cost(x, W, BV_t, BH_t, k=15)
        return x, sample, cost, monitor, params, size_bt            

    def generate_recurrence(count, k, u_tm1, primer, x, music):
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))   

#         primer = tf.zeros([1, n_visible],  tf.float32)
#         primer   = tf.slice(x, [count, 0], [1, n_visible])
        x_out, _ = RBM.gibbs_sample(primer, W, bv_t, bh_t, k=25, c_1=0.5, c_2=0.5)
    
        u_t  = (tf.tanh(bu + tf.matmul(x_out, Wvu) + tf.matmul(u_tm1, Wuu)))
        music = tf.concat(0, [music, x_out])
        return count+1, k, u_t, x_out, x, music

    def generate(num, x=x, size_bt=size_bt, u0=u0, prime_with_x=False, n_visible=n_visible, prime_length=100):
        m = tf.zeros([1, n_visible],  tf.float32)
        ct   = tf.constant(1, tf.int32) #counter
        if prime_with_x:
            Uarr = tf.scan(rnn_recurrence, x, initializer=u0)
            U = Uarr[prime_length, :, :]
        else:
            U = u0
#         x = tf.slice(x, [100, 0], [num, n_visible])
        [_, _, _, _, _, music] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                                         generate_recurrence,
                                                         [ct, tf.constant(num), U, tf.zeros([1, n_visible], tf.float32), x, m])
        return music
    return train, generate

