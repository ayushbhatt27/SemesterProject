import os
import sys
import random
import argparse
import json
import nltk
import numpy as np
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

reload(sys)
sys.setdefaultencoding('utf8')
random.seed(42)
np.random.seed(42)

DIR = "../../data/preprocessed"

def write_to_file(out_file, line):
    out_file.write(line.encode('utf8') + '\n')


def data_from_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens


def total_exs(dataset):
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total


def reporthook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner

def get_char_word_loc_mapping(context, context_tokens):
    acc = ''
    current_token_idx = 0
    mapping = dict()

    for char_idx, char in enumerate(context):
        if char != u' ' and char != u'\n':
            acc += char
            context_token = unicode(context_tokens[current_token_idx])
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (acc, current_token_idx)
                acc = ''
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def preprocess_and_write(dataset, tier, out_dir):
    num_exs = 0
    num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
    examples = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):

        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = unicode(article_paragraphs[pid]['context'])
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)
            context = context.lower()

            qas = article_paragraphs[pid]['qas']

            charloc2wordloc = get_char_word_loc_mapping(context, context_tokens)

            if charloc2wordloc is None:
                num_mappingprob += len(qas)
                continue

            for qn in qas:

                
                question = unicode(qn['question'])
                question_tokens = tokenize(question)

          			
                ans_text = unicode(qn['answers'][0]['text']).lower()
                ans_start_charloc = qn['answers'][0]['answer_start']
                ans_end_charloc = ans_start_charloc + len(ans_text)

                
                if context[ans_start_charloc:ans_end_charloc] != ans_text:
                  
                  num_spanalignprob += 1
                  continue

                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1] 
                ans_end_wordloc = charloc2wordloc[ans_end_charloc-1][1]
                assert ans_start_wordloc <= ans_end_wordloc

                
                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc+1]
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_tokenprob += 1
                    continue 

                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))

                num_exs += 1

    print ("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print ("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print ("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
    print ("Processed %i examples of total %i\n" % (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

    
    indices = range(len(examples))
    np.random.shuffle(indices)

    with open(os.path.join(out_dir, tier +'.context'), 'w') as context_file,  \
         open(os.path.join(out_dir, tier +'.question'), 'w') as question_file,\
         open(os.path.join(out_dir, tier +'.answer'), 'w') as ans_text_file, \
         open(os.path.join(out_dir, tier +'.span'), 'w') as span_file:

        for i in indices:
            (context, question, answer, answer_span) = examples[i]

            
            write_to_file(context_file, context)
            write_to_file(question_file, question)
            write_to_file(ans_text_file, answer)
            write_to_file(span_file, answer_span)


def main():
    
    train_data = data_from_json("../../data/train-v1.1.json")
    print("Train data has %i examples total" % total_exs(train_data))
    preprocess_and_write(train_data, 'train', DIR)
    dev_data = data_from_json("../../data/dev-v1.1.json")
    print("Dev data has %i examples total" % total_exs(dev_data))
    preprocess_and_write(dev_data, 'dev', DIR)


if __name__ == '__main__':
    main()
