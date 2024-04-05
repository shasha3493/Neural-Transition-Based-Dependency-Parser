#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2021-2022: Homework 3
parser_utils.py: Utilities for training the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""

import time
import os
import logging
from collections import Counter
from . general_utils import get_minibatches
from parser_transitions import minibatch_parse

from tqdm import tqdm
import torch
import numpy as np

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'


class Config(object):
    language = 'english'
    with_punct = True
    unlabeled = True
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):
        
        # iterates over eacj example in the dataset, finds the label for which head_index = 0 (implies ROOT head).
        # in dependdency parsing, label for the word whose head is ROOT is always "root"
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0]) # list of "root", 
    
        counter = Counter(root_labels) # Counter({'root': len of dataset})
        
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0] # "root"
        
        # list of "root" + all unique labels in the dataset
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))
        
        # All tokens are appended to with <l>: and id is assigned
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)
        '''
        {'<l>:root': 0, '<l>:mark': 1, '<l>:amod': 2, '<l>:det': 3, '<l>:neg': 4, '<l>:auxpass': 5, '<l>:advcl': 6, '<l>:advmod': 7, '<l>:compound': 8, '<l>:nmod': 9, '<l>:appos': 10, '<l>:det:predet': 11, '<l>:discourse': 12, '<l>:iobj': 13, '<l>:punct': 14, '<l>:acl': 15, '<l>:dobj': 16, '<l>:nmod:poss': 17, '<l>:ccomp': 18, '<l>:expl': 19, '<l>:conj': 20, '<l>:parataxis': 21, '<l>:aux': 22, '<l>:nummod': 23, '<l>:nsubj': 24, '<l>:cop': 25, '<l>:nmod:tmod': 26, '<l>:cc:preconj': 27, '<l>:nmod:npmod': 28, '<l>:case': 29, '<l>:compound:prt': 30, '<l>:dep': 31, '<l>:acl:relcl': 32, '<l>:csubj': 33, '<l>:cc': 34, '<l>:mwe': 35, '<l>:nsubjpass': 36, '<l>:xcomp': 37, '<l>:<NULL>': 38}
        '''

        config = Config()
        self.unlabeled = config.unlabeled # True
        self.with_punct = config.with_punct # True
        self.use_pos = config.use_pos # True
        self.use_dep = config.use_dep # False
        self.language = config.language # 'english'

        if self.unlabeled:
            trans = ['L', 'R', 'S']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans) # number of transitions allowed
        self.tran2id = {t: i for (i, t) in enumerate(trans)} # L: 0, R: 1, S: 2
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        # logging.info('Build dictionary for part-of-speech tags.') 
        # Appends to the tok2id dictionary. Also '<p>:<UNK>', '<p>:<NULL>', '<p>:<ROOT>' at the end
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                  offset=len(tok2id)))
        
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)
        

        # logging.info('Build dictionary for words.')
        # Appends to the tok2id dictionary. Also '<UNK>', '<NULL>', '<ROOT>' at the end
        
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                  offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)
        
        # Separate UNK, NULL token id for word, pos, label

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()} # corresponding id2tok for tok2id 
        self.n_features = 18 + (18 if config.use_pos else 0) + (12 if config.use_dep else 0) # 36
        self.n_tokens = len(tok2id) # total unique  labels of dependant + unique pos

    def vectorize(self, examples):
        '''
        For every example in the train set (i.e. for every set of words, pos, and label
        it gets the corresponding token ids and accumulates in lists. In addition it also
        prepends ROOT token id in word list, P_ROOT token id in pos, -1 in head (head of 
        ROOT is -1), -1 in label (label of root)
        
        NOTE: head is left as it. Just -1 is prepended to it
        
        NOTE: Use of on UNK token, if a particular word is not present in the vocab we use UNK token id for it
        
        For every example it adds those lists in a dictionalry with keys 'word', 'pos',
        'head', 'label'
        
        Each dictionary is apppended to a list vec_examples
        
        Eg of vec samples:
        
        {'word': [5156, 91, 113, 806, 562, 660, 88, 96, 85, 2131, 97, 109, 1001, 93, 2132, 2133, 144, 96, 2134, 2135, 361, 85, 2136, 91, 2137, 1361, 86, 97, 2138, 199, 2139, 145, 86, 85, 807, 88, 2140, 86, 808, 104, 2141, 2142, 86, 103, 1362, 1363, 89, 2143, 1364, 87], 'pos': [84, 40, 41, 42, 49, 39, 40, 61, 41, 39, 62, 40, 42, 60, 42, 42, 71, 61, 53, 44, 50, 41, 39, 40, 42, 42, 45, 62, 39, 51, 44, 72, 45, 41, 39, 40, 42, 45, 53, 40, 42, 42, 45, 48, 47, 53, 52, 42, 42, 46], 'head': [-1, 5, 5, 5, 5, 45, 9, 9, 9, 5, 9, 15, 15, 12, 15, 9, 20, 20, 19, 20, 5, 22, 20, 25, 25, 20, 20, 20, 20, 28, 28, 20, 45, 34, 45, 36, 34, 34, 34, 41, 41, 38, 34, 45, 45, 0, 48, 48, 45, 45], 'label': [-1, 18, 10, 33, 35, 6, 18, 1, 10, 6, 1, 18, 36, 18, 33, 6, 1, 1, 29, 14, 25, 10, 5, 18, 33, 6, 1, 1, 25, 24, 37, 1, 1, 10, 26, 18, 6, 1, 22, 18, 33, 6, 1, 16, 34, 0, 18, 33, 6, 1]}, {'word': [5156, 304, 1364, 1002, 2144, 87], 'pos': [84, 42, 42, 55, 42, 46], 'head': [-1, 2, 3, 0, 3, 3], 'label': [-1, 33, 14, 0, 5, 1]}]
        
        '''
        vec_examples = []
        
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']] # list of word ids + root id prepended
            
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']] # list of pos ids + p_root id prepended

            head = [-1] + ex['head'] # list of head ids + -1 prepended
            
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']] # list of label ids + -1 prepended
            
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        '''
        Given the state of stack, buffer, arcs and example in consideration find the feature set to feed to the model.
        
        Features (# list of 36 token ids)- token id for top three words in the stack, pos of top three words in the stack, token id of first three words in the buffer, pos id of (first           three words in the buffer), dependant toke id recent two left arcs for each of top two words in the stack, dependant token id recent two right arcs for each of top two words in           the stack, pos of (dependant toke id recent two left arcs for each of top two words in the stack), pos of (dependant token id recent two right arcs for each of top two words in           the stack), token id of the first dependant in the left arc of first dependant word, same for the right  and their pos 
        
        '''
        
        if stack[0] == "ROOT":
            stack[0] = 0

        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):            
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        p_features = []
        l_features = []
        
        # Use of NULL token id comes here
        
        features = [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]] # last three words (token id) in a stack. If stack has less than three words it usess NULL token id
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf)) # first three words (token id) in the buffer. if the buffer has less than 3 words it uses NULL token id
        
        if self.use_pos: # true
            p_features = [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]] # pos id of first last three words in the stack. If it has <3 words, it uses id of <P>_<NULL>
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf)) # pos id of first three words in the buffer, If it has <3 words, it uses id of <P>_<NULL>

        for i in range(2):
            # get the top two words on the stack one by one,
            if i < len(stack):
                k = stack[-i-1]
                lc = get_lc(k) # accumulate dependant words for the left arc if they exist
                rc = get_rc(k) # accumulate dependant words for the roght arc if they exist
                llc = get_lc(lc[0]) if len(lc) > 0 else [] # accumulate dependant words for the first dependant word in the left arc in its left arc if they exist
                rrc = get_rc(rc[0]) if len(rc) > 0 else [] # accumulate dependant words for the first dependant word in the right arc in its right arc if they exist
                
                # add the token id of first and second dependants for left arc and right arc (total 4 in no.)
                # add the token id of the first dpendant in the left arc of first dependant word, same for the right (2 in no.)
                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL) 
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    # add pos for the corresponding 6 tokens added above
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep: # False, not using the features from labels of dependant words
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else:
                # if not, then add corresponding 12 null token id
                features += [self.NULL] * 6
                if self.use_pos:
                    p_features += [self.P_NULL] * 6
                if self.use_dep:
                    l_features += [self.L_NULL] * 6
              
        features += p_features + l_features # add featires for pos in the featires # total 36 in nos
        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, ex):
        '''
        Given the state of stack, buffer and example in consideration find the transition step
        '''
        
        # if length of stack < 2 i.e only root present in the stack, next transition is always stack i.e. return 2
        if len(stack) < 2: 
            return self.n_trans - 1
        
        # get the top two words in the stack and their heads
        i0 = stack[-1]
        i1 = stack[-2]
        h0 = ex['head'][i0]
        h1 = ex['head'][i1]
        
        # since we are focusing on USA, labels are not used
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.unlabeled:
            # if i1 is not head and head of i1 (h1) is i0, implies left arc (with root only right arc is possible)
            if (i1 > 0) and (h1 == i0):
                return 0
            # if head of i0 (h0) is i1 and for any word in buffer its head is not i0 (if this is the case we don't do right arc bcoz if we do i0 is permamnently removed from the stack and when words are stacked, its head will never be present in the stack) implies left arc 
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return 1
            else:
                return None if len(buf) == 0 else 2 # if len buffer is 0 then return None (stop parsing)
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        '''
        examples = [{'word': [5156, 91, 113, 806, 562, 660, 88, 96, 85, 2131, 97, 109, 1001, 93, 2132, 2133, 144, 96, 2134, 2135, 361, 85, 2136, 91, 2137, 1361, 86, 97, 2138, 199, 2139, 145, 86, 85, 807, 88, 2140, 86, 808, 104, 2141, 2142, 86, 103, 1362, 1363, 89, 2143, 1364, 87], 'pos': [84, 40, 41, 42, 49, 39, 40, 61, 41, 39, 62, 40, 42, 60, 42, 42, 71, 61, 53, 44, 50, 41, 39, 40, 42, 42, 45, 62, 39, 51, 44, 72, 45, 41, 39, 40, 42, 45, 53, 40, 42, 42, 45, 48, 47, 53, 52, 42, 42, 46], 'head': [-1, 5, 5, 5, 5, 45, 9, 9, 9, 5, 9, 15, 15, 12, 15, 9, 20, 20, 19, 20, 5, 22, 20, 25, 25, 20, 20, 20, 20, 28, 28, 20, 45, 34, 45, 36, 34, 34, 34, 41, 41, 38, 34, 45, 45, 0, 48, 48, 45, 45], 'label': [-1, 18, 10, 33, 35, 6, 18, 1, 10, 6, 1, 18, 36, 18, 33, 6, 1, 1, 29, 14, 25, 10, 5, 18, 33, 6, 1, 1, 25, 24, 37, 1, 1, 10, 26, 18, 6, 1, 22, 18, 33, 6, 1, 16, 34, 0, 18, 33, 6, 1]}, {'word': [5156, 304, 1364, 1002, 2144, 87], 'pos': [84, 42, 42, 55, 42, 46], 'head': [-1, 2, 3, 0, 3, 3], 'label': [-1, 33, 14, 0, 5, 1]}]
        
        It goes over every example one by one to gets the following and add to the all_instance list
        
        1. Features for eack configuration - token id for top three words in the stack, pos of top three words in the stack, token id of first three words in the buffer, pos id of (irst            three words in the buffer), dependant toke id recent two left arcs for each of top two words in the stack, dependant token id recent two right arcs for each of top two words              in the stack, pos of (dependant toke id recent two left arcs for each of top two words in the stack), pos of (dependant token id recent two right arcs for each of top two                words in the stack), token id of the first dependant in the left arc of first dependant word, same for the right  and their pos # list of 36 token ids
        
        2. legal lables
        
        3. gold_t: transition step for each configuration
        
        Returns:
        all_instances (list of tuples) - Every tuple contains ([toled id of faetures], [label id], transition step(int)). These tuples are in tthe order of of transition steps. There is no separation between different sentences. For the entire dataset, we have list of tuples corrsoinding to (featire of the configuration at a given time, labels and next tranistion step)
        
        
        all_instances: [([5155, 5155, 5156, 91, 113, 806, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 83, 83, 84, 40, 41, 42, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83], [0, 0, 1], 2), ([5155, 5156, 91, 113, 806, 562, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 5155, 83, 84, 40, 41, 42, 49, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83], [0, 1, 1], 2)]
        
        '''
        
        
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word']) - 1 # -1 to not to consider root word # number of words in a sentence excluding the root word

            # arcs = {(h, t, label)}
            stack = [0] # initially "root" present on the stack 
            buf = [i + 1 for i in range(n_words)] # buffer, since root is taken as 0, words start from 1
            arcs = []
            instances = []
            
            # The number of steps required to parse a sentence of n words is 2n because a transition-based parser pushes n words to the stack with SHIFT and then removes those n words with the LEFT- or RIGHT-ARC
            
            for i in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex) # gold transition = 0 for left arc, 1 for right arc, 2 for stack
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf) # [0, 0, 1]
                
                
                assert legal_labels[gold_t] == 1
                # for every example in the train set, (features, legal_labels, gold_t) is appended to instances list
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))

                
                # stack -> push word to the stack and remove the word from buffer
                if gold_t == self.n_trans - 1:
                    stack.append(buf[0])
                    buf = buf[1:]
                    
                # left arc - appends the arc in arcs list, remove second word from the stack
                elif gold_t < self.n_deprel:
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack = stack[:-2] + [stack[-1]]
                    
                # right arc - appends the arc in arcs list, remove first word from the stack
                else:
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                    stack = stack[:-1]
            else:
                # once a sentence is finished, add the list of tuples to the all instance list.
                succ += 1
                all_instances += instances # every list instance in all instances corresponds to a sentence. Every list inside all instances has tuples as its elemenets. Each tuple reprents each featires, legal_labels and transition step when parsing the sentence
                
        return all_instances

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel # self.n_deprel = 1
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels

    def parse(self, dataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
              
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)] # index words from 1 to len(words)-1 for a sentence
            sentences.append(sentence) # list of list.
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx) # INitialize model parser
        dependencies = minibatch_parse(sentences, model, eval_batch_size) # list of list of tuples, internal list of tuples represents dependencies for each setence
        UAS = all_tokens = 0.0
        # iterate over every setence, and match the predicted head of each word to that of groudn truth. 
        # For eack token match, we increase the score by 1. Finally UAS is calculated as (total score/total token in the val set)
        with tqdm(total=len(dataset)) as prog:
            for i, ex in enumerate(dataset):
                head = [-1] * len(ex['word'])
                for h, t, in dependencies[i]:
                    head[t] = h
                for pred_h, gold_h, gold_l, pos in \
                        zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                        assert self.id2tok[pos].startswith(P_PREFIX)
                        pos_str = self.id2tok[pos][len(P_PREFIX):]
                        if (self.with_punct) or (not punct(self.language, pos_str)):
                            UAS += 1 if pred_h == gold_h else 0
                            all_tokens += 1
                prog.update(i + 1)
        UAS /= all_tokens
        return UAS, dependencies


class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        # same as training, extract features and feed to the model to get the predictions
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32') # (bs, 36)
        mb_x = torch.from_numpy(mb_x).long()
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]

        pred = self.parser.model(mb_x) # (bs, 3)
        pred = pred.detach().numpy()
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'), 1)
        pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred] # (bs, 3)
        return pred


def read_conll(in_file, lowercase=False, max_example=None):
    '''
    It uses Penn Treebank dataset.
    
    We have a separate file for train and dev (val) and test.
    
    In a file, every row corresponds to a word in a sentence. 
    Details of that word are separated by a tab
    
    index    Word    _    word_type    POS    _    index_of_head_word    label_of_determinant
    
    Alphabetical list of part-of-speech tags used in the Penn Treebank Project: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    
    Example:
    Sentence: A record data hasn't been set.

    1	A	_	DET	DT	_	3	det	_	_
    2	record	_	NOUN	NN	_	3	compound	_	_
    3	date	_	NOUN	NN	_	7	nsubjpass	_	_
    4	has	_	AUX	VBZ	_	7	aux	_	_
    5	n't	_	PART	RB	_	7	neg	_	_
    6	been	_	AUX	VBN	_	7	auxpass	_	_
    7	set	_	VERB	VBN	_	0	root	_	_
    8	.	_	PUNCT	.	_	7	punct	_	_
    
    For every new sentence it's seperate by a blank row
    
    '''
    
    # Read the words details for every sentence one by one. 
    # For every setence, collect words, pos, head, label in a a list.
    # After every sentence, create a dictionary - {'word': word, 'pos': pos, 'head': head, 'label': label}
    # and append to the examples list
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]: # if False it imples a new sentence
                    word.append(sp[1].lower() if lowercase else sp[1]) #lower case the word
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                # create a dictionary of word, pos, head and label and append to the examples list
                # {'word': ['ms.', 'haag', 'plays', 'elianti', '.'], 'pos': ['NNP', 'NNP', 'VBZ', 'NNP', '.'], 'head': [2, 3, 0, 3, 3], 'label': ['compound', 'nsubj', 'root', 'dobj', 'punct']}
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    
    '''
[{'word': ['in', 'an', 'oct.', '19', 'review', 'of', '``', 'the', 'misanthrope', "''", 'at', 'chicago', "'s", 'goodman', 'theatre', '-lrb-', '``', 'revitalized', 'classics',            'take', 'the', 'stage', 'in', 'windy', 'city', ',', "''", 'leisure', '&', 'arts', '-rrb-', ',', 'the', 'role', 'of', 'celimene', ',', 'played', 'by', 'kim', 'cattrall', ',', 'was',   'mistakenly', 'attributed', 'to', 'christina', 'haag', '.'], 'pos': ['IN', 'DT', 'NNP', 'CD', 'NN', 'IN', '``', 'DT', 'NN', "''", 'IN', 'NNP', 'POS', 'NNP', 'NNP', '-LRB-', '``', 'VBN', 'NNS', 'VB', 'DT', 'NN', 'IN', 'NNP', 'NNP', ',', "''", 'NN', 'CC', 'NNS', '-RRB-', ',', 'DT', 'NN', 'IN', 'NNP', ',', 'VBN', 'IN', 'NNP', 'NNP', ',', 'VBD', 'RB', 'VBN', 'TO', 'NNP', 'NNP', '.'], 'head': [5, 5, 5, 5, 45, 9, 9, 9, 5, 9, 15, 15, 12, 15, 9, 20, 20, 19, 20, 5, 22, 20, 25, 25, 20, 20, 20, 20, 28, 28, 20, 45, 34, 45, 36, 34, 34, 34, 41, 41, 38, 34, 45, 45, 0, 48, 48, 45, 45], 'label': ['case', 'det', 'compound', 'nummod', 'nmod', 'case', 'punct', 'det', 'nmod', 'punct', 'case', 'nmod:poss', 'case', 'compound', 'nmod', 'punct', 'punct', 'amod', 'nsubj', 'dep', 'det', 'dobj', 'case', 'compound', 'nmod', 'punct', 'punct', 'dep', 'cc', 'conj', 'punct', 'punct', 'det', 'nsubjpass', 'case', 'nmod', 'punct', 'acl', 'case', 'compound', 'nmod', 'punct', 'auxpass', 'advmod', 'root', 'case', 'compound', 'nmod', 'punct']}, {'word': ['ms.', 'haag', 'plays', 'elianti', '.'], 'pos': ['NNP', 'NNP', 'VBZ', 'NNP', '.'], 'head': [2, 3, 0, 3, 3], 'label': ['compound', 'nsubj', 'root', 'dobj', 'punct']}, {'word': ['rolls-royce', 'motor', 'cars', 'inc.', 'said', 'it', 'expects', 'its', 'u.s.', 'sales', 'to', 'remain', 'steady', 'at', 'about', '1,200', 'cars', 'in', '1990', '.'], 'pos': ['NNP', 'NNP', 'NNPS', 'NNP', 'VBD', 'PRP', 'VBZ', 'PRP$', 'NNP', 'NNS', 'TO', 'VB', 'JJ', 'IN', 'IN', 'CD', 'NNS', 'IN', 'CD', '.'], 'head': [4, 4, 4, 5, 0, 7, 5, 10, 10, 7, 12, 7, 12, 17, 16, 17, 12, 19, 12, 5], 'label': ['compound', 'compound', 'compound', 'nsubj', 'root', 'nsubj', 'ccomp', 'nmod:poss', 'compound', 'dobj', 'mark', 'xcomp', 'xcomp', 'case', 'advmod', 'nummod', 'nmod', 'case', 'nmod', 'punct']}]
    '''
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)
    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        # http://nlp.stanford.edu/software/spanish-faq.shtml
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)


def minibatches(data, batch_size):
    '''
    Accummulate features and labels from the entire dataset. Further one hot encode the labels 
    '''
    x = np.array([d[0] for d in data]) # get features in the entire train data # (2*no. of words in the corpus, 36) (As it takes 2*n_words steps to complete parsing)
    y = np.array([d[2] for d in data]) # transiton step (2*no.of words) 
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1 # one hot encoding of labels (2*no.of words, 3) 
    return get_minibatches([x, one_hot], batch_size)


def load_and_preprocess_data(reduced=True):
    config = Config()
    print("Loading data...",)
    start = time.time()
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    
    # if debug mode is on, only take a subset of train, dev and test samples
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...",)
    start = time.time()
    parser = Parser(train_set) # initialize the Parser class with train set
    print("took {:.2f} seconds".format(time.time() - start))

    print("Loading pretrained embeddings...",)
    start = time.time()
    
    '''
    ./data/en-cw.txt is at text file. Eevry line in the text file represents a pretrained word embedding of size 50 for pos, labels of deteminant and words. 
    For a line, each element is separted by a space. First element is the token and from 2nd to 51st elements are the values of 
    word vector.
    '''
    word_vectors = {} # a dictionary of word vectors + posin the vocabulary
    
    for line in open(config.embedding_file).readlines(): # './data/en-cw.txt'
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]

    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32') # initialize embedding matrix from a normal distribution
    
    # go to very token in tok2id, if the token is present in word_vectors
    # use the pretrained embedding to initialize the row correspoing to the id of the token
    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print("took {:.2f} seconds".format(time.time() - start))
    # embedding matrix is a 2d array of shape (n_tokens, 50). 
    # Row i of the embedding matrix represents word vector for a the token whose id = i
    
    print("Vectorizing data...",)
    start = time.time()
    # Note we only use train data to prepare vocab for words,pos, labels. For dev and test set, if something is not present in vocab we use UNK token id for it
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))
    
    
    print("Preprocessing training data...",)
    start = time.time()
    train_examples = parser.create_instances(train_set)
    print("took {:.2f} seconds".format(time.time() - start))

    return parser, embeddings_matrix, train_examples, dev_set, test_set,


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    pass
