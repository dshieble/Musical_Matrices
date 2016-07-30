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


#clip gradients
#don't compute gradients on sample


train, generate = rnn_rbm.build_rnnrbm()
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
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
gvs = optimizer.compute_gradients(cost, [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu])
gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
train_var = optimizer.apply_gradients(gvs)
saver = tf.train.Saver()


num_epochs = 200
batch_size = 100
NUM_CORES = 4

# songs, dt, r = get_songs('data/music_all/train/*.mid')
songs, dt, r = rnn_rbm.get_songs('data/Nottingham/train_C/*.mid', 1)



with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES)) as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    merged = tf.merge_all_summaries()
    os.system("rm -rf /tmp/tf_rbm_logs/train")
    writer = tf.train.SummaryWriter('/tmp/tf_rbm_logs/train', sess.graph)

    # loop with batch
    for epoch in range(num_epochs):
        for song in songs:
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                alpha = min(0.05, 100/float(i))
#                 print "iteration: {}, shape: {}".format(i, tr_x.shape)
                summary, _ = sess.run([merged, train_var], feed_dict={x: tr_x, a: alpha})
                writer.add_summary(summary, i)
        print "epoch: {} cost: {}".format(epoch, cost.eval(session=sess, feed_dict={x: tr_x, a: alpha}))
        if (epoch + 1) % 10 == 0:
            save_path = saver.save(sess, "saved_data/rbm_rnn_epoch_{}".format(epoch))

            for i in range(10):
                generated_music = sess.run(generate())
                midiwrite("music_outputs/tf_training_iter/epoch_{}_composition_tf_{}.midi".format(epoch, i), 
                          generated_music, r, dt)

