from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys
from collections import Counter
import string
import re
import argparse
import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from fetch_data import get_batch_generator
from utilities import  *

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()

    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def exact_match_score(prediction, ground_truth):
        return (normalize_answer(prediction) == normalize_answer(ground_truth))


    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


    def evaluate(dataset, predictions):
        f1 = exact_match = total = 0
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                  ' will receive score 0.'
                        #print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths)
                    f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {'exact_match': exact_match, 'f1': f1}

    def print_example(word2id, context_tokens, qn_tokens, true_ans_start, true_ans_end, pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em):
        curr_context_len = len(context_tokens)
        context_tokens = [w if w in word2id else "_%s_" % w for w in context_tokens]
        truncated = False
        for loc in range(true_ans_start, true_ans_end+1):
            if loc in range(curr_context_len):
                context_tokens[loc] = context_tokens[loc]
            else:
                truncated = True
        assert pred_ans_start in range(curr_context_len)
        assert pred_ans_end in range(curr_context_len)

        print("".join(context_tokens))
        question = " ".join(qn_tokens)

        print ("{:>20}: {}".format("QUESTION", question))
        if truncated:
            print ("{:>20}: {}".format("TRUE ANSWER", true_answer))
            print ("{:>22}(True answer was truncated from context)".format(""))
        else:
            print ("{:>20}: {}".format("TRUE ANSWER", true_answer))
        print ("{:>20}: {}".format("PREDICTED ANSWER", pred_answer))

    def add_placeholders(self):
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        with vs.variable_scope("embeddings"):
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix")
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids)

    def build_graph(self):
        encoder = RNNEncoder(self.FLAGS.hidden_size_encoder, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask, scopename='RNNEncoder')
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask, scopename='RNNEncoder')

        last_dim = context_hiddens.get_shape().as_list()[-1]

        attn_layer = BasicAttn(self.keep_prob, last_dim, last_dim)
        _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens)  

        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) 

        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size_fully_connected)

        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


    def add_loss(self):
        with vs.variable_scope("loss"):

            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0])
            self.loss_start = tf.reduce_mean(loss_start)
            tf.summary.scalar('loss_start', self.loss_start)

            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        
        start_dist, end_dist = self.get_prob_dists(session, batch)
        maxprob = 0
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)
        return start_pos, end_pos, maxprob

    def get_attention_dist(self, session, batch):

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        if self.FLAGS.do_char_embed:
            input_feed[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
            input_feed[self.char_ids_qn] = self.padded_char_ids(batch, batch.qn_ids)
        
        
        if self.FLAGS.rnet_attention:
            output_feed = [self.rnet_attention_probs]
        
        elif self.FLAGS.bidaf_attention:
            output_feed = [self.bidaf_attention_probs]
        
        [attn_distribution] = session.run(output_feed, input_feed)

        start_dist, end_dist = self.get_prob_dists(session, batch)

        return start_dist

    def matrix_multiplication(self, mat, weight):

        mat_shape = mat.get_shape().as_list()  # shape - ijk
        weight_shape = weight.get_shape().as_list()  # shape -kl
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])  # reshape to batch_size, seq_len, p

    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print ("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic))

        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos, _ = self.get_start_end_pos(session, batch)

            pred_start_pos = pred_start_pos.tolist()
            pred_end_pos = pred_end_pos.tolist()
            print ("*****************************************************************")

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                true_answer = " ".join(true_ans_tokens)

                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        exp_loss = None

        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                if not exp_loss:
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                if global_step % self.FLAGS.eval_every == 0:

                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)