import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

class RNNEncoder(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks, scopename):
        with vs.variable_scope(scopename):
            input_lens = tf.reduce_sum(masks, reduction_indices=1)
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            out = tf.concat([fw_out, bw_out], 2)
            out = tf.nn.dropout(out, self.keep_prob)
            return out

class SimpleSoftmaxLayer(object):
    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        with vs.variable_scope("SimpleSoftmaxLayer"):

            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None)
            logits = tf.squeeze(logits, axis=[2])

            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        with vs.variable_scope("BasicAttn"):
            values_t = tf.transpose(values, perm=[0, 2, 1])
            attn_logits = tf.matmul(keys, values_t)
            print("Basic attn keys", keys.shape)
            print("Basic attn values", values_t.shape)
            print("Basic attn logits", attn_logits.shape)
            attn_logits_mask = tf.expand_dims(values_mask, 1)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2)
            output = tf.matmul(attn_dist, values)
            output = tf.nn.dropout(output, self.keep_prob)
            return attn_dist, output

def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)
    masked_logits = tf.add(logits, exp_mask)
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
