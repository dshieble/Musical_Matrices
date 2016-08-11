import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import pandas as pd


# helper functions
def sample(probs, binary=True, c_1=0.5, c_2=0.5):
    sh = tf.shape(probs)[1]/2
    adder = tf.concat(0, [tf.ones([sh], tf.float32)*(0.5-c_1), tf.ones([sh], tf.float32)*(0.5-c_1)])
    #SET TO BE GAUSSIAN IF NOT BINARY
    sampled = probs + tf.random_uniform(tf.shape(probs), 0, 1) + adder
    return (sampled/2) if not binary else tf.floor(sampled)

def gibbs_forward(xk, W, bh, binary=True):
    return sample(tf.sigmoid(tf.matmul(xk, W) + bh), binary=binary)

def gibbs_sample(x, W, bv, bh, k, binary=True, c_1=0.5, c_2=0.5):

    size_bt = tf.shape(x)[0]
    # CD-k
    # we use tf.while_loop to achieve the multiple (k - 1) gibbs sampling  
    def gibbs_step(count, k, xk, raw_xk, W=W):
        hk = gibbs_forward(xk, W, bh, binary)
        C1 = tf.cond(count + 1 >= k, lambda: tf.constant(c_1), lambda: tf.constant(0.5))
        C2 = tf.cond(count + 1 >= k, lambda: tf.constant(c_2), lambda: tf.constant(0.5))
        raw_xk = tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)
        xk = sample(raw_xk, binary=binary, c_1=c_1, c_2=c_2)
        return count+1, k, xk, raw_xk

    ct = tf.constant(0) #counter
    h = sample(tf.sigmoid(tf.matmul(x, W) + bh), binary=binary)
    [_, _, xk1, raw_xk] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x, x], 1, False)
    xk1 = tf.stop_gradient(xk1)
    raw_xk = tf.stop_gradient(raw_xk)
    return xk1, raw_xk

def get_free_energy_cost(x, W, bv, bh, k, binary=True):
    size_bt = tf.shape(x)[0]
    
    # define graph/algorithm
    xk1, _    = gibbs_sample(x,   W, bv, bh, k, binary=binary)

    def free_energy(xx):
        #return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
        lin_out = tf.matmul(xx, W) + bh
        A = -tf.reduce_sum(tf.log(1 + tf.exp(lin_out)), 1)
        B = -tf.matmul(xx, tf.transpose(bv))
        return tf.add(B, A)

    #Define loss and optimizer
    #the cost is based on the difference in free energy between v and v_sample
    cost = tf.reduce_mean(tf.sub(free_energy(x), free_energy(xk1)))
    
    #The Pseudo Log Likelihood Monitor. This is only used for tracking learning
    #T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    _, raw_xk = gibbs_sample(xk1, W, bv, bh, 1, binary=binary)
    raw_xk    = raw_xk[0, :]
    monitor = tf.reduce_mean(x*tf.log(raw_xk) + (1 - x)*tf.log(1 - raw_xk))

    return xk1, cost, monitor

def get_cd_update(x, W, bv, bh, k, lr, binary=True):
    #Get the contrastive divergence update variable
    lr = tf.constant(lr, tf.float32)
    size_bt = tf.cast(tf.shape(x)[0], tf.float32)
                                                
    # define graph/algorithm
    h = gibbs_forward(x, W, bh, binary)
    xk1, raw_xk = gibbs_sample(x, W, bv, bh, k, binary=binary)
    hk1 = gibbs_forward(xk1, W, bh, binary)
    # update rule
    W_  = tf.mul(lr/size_bt, tf.sub(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(xk1), hk1)))
    bv_ = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(x, xk1), 0, True))
    bh_ = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(h, hk1), 0, True))

    # wrap session
    updt = [W.assign_add(W_), bv.assign_add(bv_), bh.assign_add(bh_)]
    return updt

