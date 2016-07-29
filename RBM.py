import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import pandas as pd

# helper functions
def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def gibbs_sample(x, W, bv_t, bh_t, k):
    size_bt = tf.shape(x)[0]
    # CD-k
    # we use tf.while_loop to achieve the multiple (k - 1) gibbs sampling  
    # set up tf.while_loop()
    def rbmGibbs(xk, count, k, W=W):
#         xk = tf.Print(xk, [xk], "xk start ")
        hk = sampleInt(tf.sigmoid(tf.matmul(xk, W) + tf.tile(bh_t, [size_bt, 1])))
        xk = sampleInt(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + tf.tile(bv_t, [size_bt, 1])))
#         xk = tf.Print(xk, [xk], "xk end ")
        return xk, count+1, k

    def lessThanK(xk, count, k):
        return count <= k

    ct = tf.constant(1) #counter
    [xk1, _, _] = control_flow_ops.While(lessThanK, rbmGibbs, [x, ct, tf.constant(k)], 1, False)
    return xk1

def build_rbm(x, W, bv_t, bh_t, k):
#     W    = tf.Print(W, [W], "W")
#     bv_t    = tf.Print(bv_t, [bv_t], "bv_t")
#     bh_t    = tf.Print(bh_t, [bh_t], "bh_t")

    size_bt = tf.shape(x)[0]
    
    # define graph/algorithm
    xk1 = gibbs_sample(x, W, bv_t, bh_t, k)
    xk1 = tf.stop_gradient(xk1)

    def free_energy(xx):
        #return -(v * bv_t).sum() - T.log(1 + T.exp(T.dot(v, W) + bh_t)).sum()
        lin_out = tf.matmul(xx, W) + tf.tile(bh_t, [size_bt, 1])
        A = -tf.reduce_sum(tf.log(1 + tf.exp(lin_out)), 1)
        B = -tf.matmul(xx, tf.transpose(tf.cast(tf.tile(bv_t, [size_bt, 1]), tf.float32)))
#         out =  tf.matmul(xx, W) + tf.tile(bh_t, [size_bt, 1])
#         out = tf.matmul(xx, tf.transpose(tf.cast(tf.tile(bv_t, [size_bt, 1]), tf.float32)))
#         out    = tf.Print(out, [out], "out")
#         out    = tf.Print(out, [tf.reduce_any(tf.equal(out, tf.zeros(tf.shape(out))))],
#                           "tf.equal(out, tf.zeros(tf.shape(out)))]")
#         return tf.matmul(xx, 
#     
#         return tf.clip_by_value(lin_out,1e-10,1.0)#
        return tf.add(B, A)

    #Define loss and optimizer
    #the cost is based on the difference in free energy between v and v_sample
    cost = tf.sub(tf.reduce_mean(free_energy(x)), tf.reduce_mean(free_energy(xk1)))
#     cost    = tf.Print(cost, [e1], "e1")
#     cost    = tf.Print(cost, [e2], "e2")
#     cost    = tf.Print(cost, [tf.reduce_any(tf.is_nan(e1))], "tf.reduce_any(tf.is_nan(e1))")
#     cost    = tf.Print(cost, [tf.reduce_any(tf.is_nan(e2))], "tf.reduce_any(tf.is_nan(e2))")
#     cost    = tf.Print(cost, [cost], "cost")
    return xk1, cost
