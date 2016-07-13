import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

# Parameters

# Network Parameters

def RNN(x, weights, biases, n_input, n_steps, n_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    linear_out = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.sigmoid(linear_out)

def get_forecast_Xy(L, timestep):
    outX = [L[i-timestep:i] for i in range(timestep, len(L))]
    outy = [L[i]            for i in range(timestep, len(L))]  
    return np.array(outX), np.array(outy)

def compose(x, sess, pred, primer, steps=100):

    predictions = primer
    for i in tqdm(range(steps)):
        primer = predictions[None, -200:, :]
        RP = sess.run(pred, feed_dict={x: primer})
        P = RP > 0.9
        predictions = np.vstack((predictions, P))

    return predictions[200:, :]


# def get_forecast_batch(L, timestep, batch_size=100):
#     for i in range(timestep, len(L), batch_size):
#         outX = [L[i-timestep:i] for i in range(batch_size)]
#         outy = [L[i]            for i in range(batch_size)]
#         yield np.array(outX), np.array(outy)
