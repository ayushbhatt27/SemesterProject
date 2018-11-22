from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging

import tensorflow as tf

from layer_structure import QAModel
from wordvec_loader import get_glove

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") 
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") 

tf.app.flags.DEFINE_integer("gpu", 0, "number of gpu's")
tf.app.flags.DEFINE_string("experiment_name", "basic_model", "Unique name for your experiment.")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train.")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "for regularization")
tf.app.flags.DEFINE_integer("batch_size", 60, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size_encoder", 150, "Size of the hidden states")
tf.app.flags.DEFINE_integer("hidden_size_fully_connected", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 300, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors.")

tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep.")

tf.app.flags.DEFINE_string("train_dir", "", "Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Defaults to predictions.json")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir, expect_exists):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
            session.run(tf.global_variables_initializer())
            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())


def main(unused_argv):
    experiment_name = "bidaf_best"

    bestmodel_dir = "../experiments/bidaf_best/best_checkpoint"

    glove_path = "../data/glove.6B.100d.txt"

    emb_matrix, word2id, id2word = get_glove(glove_path, 100)

    manual_context = "../data/dev.context"
    manual_question = "../data/dev.question"
    manual_span = "../data/dev.span"

    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

        f1, em = qa_model.check_f1_em(sess, manual_context, manual_question, manual_span, "dev", num_samples=10, print_to_screen=True)
        print 'Accuracy: %s' % f1 


if __name__ == "__main__":
    tf.app.run()
