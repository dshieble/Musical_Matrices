import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

# Parameters

# Network Parameters
def create_cell(input_size, output_size, dropout=False, cell_type="lstm"):
    if cell_type == "vanilla":
        cell_class = tf.nn.rnn_cell.BasicRNNCell
    elif cell_type == "gru":
        cell_class = tf.nn.rnn_cell.BasicGRUCell
    elif cell_type == "lstm":
        cell_class = tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("Invalid cell type: {}".format(cell_type))

    cell = cell_class(output_size, input_size = input_size)
    if dropout:
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.5)
    else:
        return cell

            
def RNN(x, weights, biases, n_input, n_steps, hidden_sizes):

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

    cell = tf.nn.rnn_cell.MultiRNNCell(
            [create_cell(hidden_sizes[0])] + [create_cell(s) for s in hidden_sizes[1:] + [128]])

    
    outputs, states = tf.nn.rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    linear_out = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.sigmoid(linear_out)

def get_forecast_Xy(L, timestep):
    outX = [L[i-timestep:i] for i in range(timestep, len(L))]
    outy = [L[i]            for i in range(timestep, len(L))]  
    return np.array(outX), np.array(outy)

def compose(x, sess, pred, primer, steps=100, timestep=200):

    predictions = primer
    for i in tqdm(range(steps)):
        primer = predictions[None, -timestep:, :]
        RP = sess.run(pred, feed_dict={x: primer})
        P = RP > 0.9999
        predictions = np.vstack((predictions, P))

    return predictions[timestep:, :]


# def get_forecast_batch(L, timestep, batch_size=100):
#     for i in range(timestep, len(L), batch_size):
#         outX = [L[i-timestep:i] for i in range(batch_size)]
#         outy = [L[i]            for i in range(batch_size)]
#         yield np.array(outX), np.array(outy)
