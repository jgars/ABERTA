import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import itertools as it
import re


def load_file(file_path, level_list=(13, 26,), punctuation_list=(',', '.',)):
    text_file = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        sentence = lines[0]
        original_sentence, sentence = preprocess_text(sentence, level_list=level_list,
                                                      punctuation_list=punctuation_list)
        text_file['text'] = sentence
        text_file['original_text'] = original_sentence
    return text_file


def preprocess_text(sentence, level_list=(5, 10, 30), punctuation_list=(',', '.', '...')):
    sentence = sentence.strip().replace(" ", "").replace('<s>', '-')
    sentence = sentence.replace("--------------",
                                "-------------|")
    original_sentence = sentence
    new_sentence = []
    for g in it.groupby(list(sentence)):
        g_list = list(g[1])
        if g[0] != '|' and g[0] != '-':
            new_sentence.append(g[0])
        else:
            new_sentence.extend(g_list)
    sentence = ''.join(new_sentence)

    # Remove blank token in a word
    sentence = re.sub('\\b(-)+\\b', '', sentence)

    # Deal with "'" token
    sentence = re.sub('\\b(-)+\'', '\'', sentence)
    sentence = re.sub('\'(-)+\\b', '\'', sentence)

    sentence = sentence.replace("-", "|")
    while sentence.startswith('|'):
        sentence = sentence[1:]
    assert len(level_list) == len(punctuation_list), 'level_list and punctuation_list must have the same length.'
    for level, punctuation in zip(reversed(level_list), reversed(punctuation_list)):
        sentence = re.sub('\|{' + str(level) + ',}', punctuation + ' ', sentence)
    sentence = re.sub('\|+', ' ', sentence)
    return original_sentence, sentence
