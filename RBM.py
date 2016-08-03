import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import pandas as pd

# helper functions
def sample(probs, binary=True):
    sampled = probs + tf.random_uniform(tf.shape(probs), 0, 1)
    return sampled if not binary else tf.floor(sampled)

def gibbs_sample(x, W, bv_t, bh_t, k, binary=True):
    size_bt = tf.shape(x)[0]
    # CD-k
    # we use tf.while_loop to achieve the multiple (k - 1) gibbs sampling  
    def rbmGibbs(count, k, xk, W=W):
#         xk = tf.Print(xk, [xk], "xk start ")
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh_t), binary=binary)#tf.tile(bh_t, [size_bt, 1])))
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv_t), binary=binary)#tf.tile(bv_t, [size_bt, 1])))
#         xk = tf.Print(xk, [xk], "xk end ")
        return count+1, k, xk

    ct = tf.constant(0) #counter
    [_, _, xk1] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                         rbmGibbs, [ct, tf.constant(k), x], 1, False)
    xk1 = tf.stop_gradient(xk1)
    return xk1

def build_rbm(x, W, bv_t, bh_t, k, binary=True):
#     W    = tf.Print(W, [W], "W")
#     bv_t    = tf.Print(bv_t, [bv_t], "bv_t")
#     bh_t    = tf.Print(bh_t, [bh_t], "bh_t")

    size_bt = tf.shape(x)[0]
    
    # define graph/algorithm
    xk1 = gibbs_sample(x, W, bv_t, bh_t, k, binary=binary)
    xk1 = tf.stop_gradient(xk1)

    def free_energy(xx):
        #return -(v * bv_t).sum() - T.log(1 + T.exp(T.dot(v, W) + bh_t)).sum()
        lin_out = tf.matmul(xx, W) + bh_t
        A = -tf.reduce_sum(tf.log(1 + tf.exp(lin_out)), 1)
        B = -tf.matmul(xx, tf.transpose(bv_t))
#         A = tf.Print(A, [tf.shape(A)], "tf.shape(A)")
#         B = tf.Print(B, [tf.shape(B)], "tf.shape(B)")
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
    cost = tf.reduce_mean(tf.sub(free_energy(x), free_energy(xk1)))
#     cost    = tf.Print(cost, [e1], "e1")
#     cost    = tf.Print(cost, [e2], "e2")
#     cost    = tf.Print(cost, [tf.reduce_any(tf.is_nan(e1))], "tf.reduce_any(tf.is_nan(e1))")
#     cost    = tf.Print(cost, [tf.reduce_any(tf.is_nan(e2))], "tf.reduce_any(tf.is_nan(e2))")
#     cost    = tf.Print(cost, [cost], "cost")
    return xk1, cost
