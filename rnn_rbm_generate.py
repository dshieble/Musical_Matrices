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
import time

def main(saved_weights_path, num, primer_path):
    train, generate = rnn_rbm.build_rnnrbm()
    x, sample, cost, params, size_bt = train()
    W, bh, bv, x, a, Wuh, Wuv, Wvu, Wuu, bu, u0 = params

    saver = tf.train.Saver()

    song_primer = rnn_rbm.get_song(primer_path)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver.restore(sess, saved_weights_path)
        for i in tqdm(range(num)):
            generated_music = sess.run(generate(300, prime_with_x=True), feed_dict={x: song_primer})
            save_path = "music_outputs/generated_music/{}_{}".format(i, primer_path.split("/")[-1])
            rnn_rbm.write_song(save_path, generated_music)
            
if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    
# python rnn_rbm_generate.py  saved_data/rbm_rnn_epoch_19 10 data/assorted_classical/classical_piano_C/beethoven_hammerklavier_2_C.mid