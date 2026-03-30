import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num
from core import load_file
import re
import itertools as it
import numpy as np
from itertools import combinations
import torch
import transformers
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_text(sentence):
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
    sentence = re.sub('\\b(-)+\\b', '', sentence)
    sentence = re.sub('\\b(-)+\'', '\'', sentence)
    sentence = re.sub('\'(-)+\\b', '\'', sentence)
    sentence = sentence.replace("-", "|")
    while sentence.startswith('|'):
        sentence = sentence[1:]
    return sentence.lower()


def get_no_punctuation_text(sentence):
    sentence = re.sub('\|+', ' ', sentence)
    return sentence


def count_pause(sentence):
    idx = 0
    cnt_list = []
    while idx < len(sentence):
        while idx < len(sentence) and sentence[idx] != '|':
            idx += 1
        cnt = 0
        while idx < len(sentence) and sentence[idx] == '|':
            idx += 1
            cnt += 1
        cnt_list.append(cnt)
    return cnt_list


def process_path(path):
    all_cnt_list = []
    all_text_list = []
    file_list = [name for name in sorted(os.listdir(path)) if name.endswith('.txt')]
    for name in file_list:
        file_path = os.path.join(path, name)
        text = load_file(file_path, level_list=(), punctuation_list=())
        text['text'] = text['text'].lower()
        print(file_path, len(text['text'].split()))
        print(text['text'])
        text['original_text'] = preprocess_text(text['original_text'])
        text['no_punctuation_text'] = get_no_punctuation_text(text['original_text'])
        file_cnt_list = count_pause(text['original_text'])
        if len(text['text'].split()) > 20:
            all_cnt_list.extend(file_cnt_list)
            all_text_list.append(text)
    return all_cnt_list, all_text_list


def find_threshold(hc_cnt_list, hc_text_list, ad_cnt_list, ad_text_list, sentence_threshold):
    max_number = np.maximum(np.max(hc_cnt_list), np.max(ad_cnt_list)) + 1
    hc_bin_cnt = np.bincount(hc_cnt_list, minlength=max_number)
    print(hc_bin_cnt)
    ad_bin_cnt = np.bincount(ad_cnt_list, minlength=max_number)
    print(ad_bin_cnt)

    hc_bin_cnt = hc_bin_cnt[:sentence_threshold]
    ad_bin_cnt = ad_bin_cnt[:sentence_threshold]

    level = 1
    comb_list = combinations(range(2, sentence_threshold), level)
    comb_list = list(comb_list)

    max_sum_diff = 0
    best_comb = []
    for comb in comb_list:
        comb = list(comb)
        comb.append(sentence_threshold)
        pre_level = 0
        sum_diff = 0
        for i in range(level + 1):
            sum_diff += np.abs(np.sum(hc_bin_cnt[pre_level:comb[i] + 1] / len(hc_text_list)) -
                               np.sum(ad_bin_cnt[pre_level:comb[i] + 1]) / len(ad_text_list))
            pre_level = comb[i] + 1
        if sum_diff > max_sum_diff:
            best_comb = comb
            max_sum_diff = sum_diff
    print(best_comb)


def find_entropy_threshold(hc_text_list, ad_text_list):
    model_name = 'bert-base-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name, model_max_length=512)
    model = transformers.BertForPreTraining.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def get_entropy_score(sentence):
        inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        inputs.to(device)
        token_list = tokenizer.tokenize(sentence)
        predictions = model(**inputs)
        predictions = predictions['prediction_logits']
        label = inputs["input_ids"].squeeze()
        for idx, token in enumerate(token_list[:int(tokenizer.model_max_length - 1)]):
            if token in [',', '.', ';']:
                label[idx + 1] = -100
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(), label).data
        return float(loss)

    min_epoch_loss_entropy = 0.
    best_period_threshold = None
    for period_threshold in range(1, 1000):
        epoch_loss_classification_ad = []
        epoch_loss_classification_hc = []
        for text in ad_text_list:
            sentence = text['original_text']
            sentence = re.sub('\|{' + str(period_threshold) + ',}', '. ', sentence)
            sentence = re.sub('\|+', ' ', sentence)
            epoch_loss_classification_ad.append(get_entropy_score(sentence))
        for text in hc_text_list:
            sentence = text['original_text']
            sentence = re.sub('\|{' + str(period_threshold) + ',}', '. ', sentence)
            sentence = re.sub('\|+', ' ', sentence)
            epoch_loss_classification_hc.append(get_entropy_score(sentence))
        epoch_loss_classification_diff = \
            abs(np.median(epoch_loss_classification_ad) - np.median(epoch_loss_classification_hc))
        if epoch_loss_classification_diff > min_epoch_loss_entropy:
            min_epoch_loss_entropy = epoch_loss_classification_diff
            best_period_threshold = period_threshold
            print(min_epoch_loss_entropy, best_period_threshold)
    print(best_period_threshold)

    return best_period_threshold


def main():
    full_wave_enhanced_audio_path = 'data/adresso21/diagnosis/train/asr_text/cn'
    hc_cnt_list, hc_text_list = process_path(full_wave_enhanced_audio_path)
    full_wave_enhanced_audio_path = 'data/adresso21/diagnosis/train/asr_text/ad'
    ad_cnt_list, ad_text_list = process_path(full_wave_enhanced_audio_path)

    best_period_threshold = find_entropy_threshold(hc_text_list, ad_text_list)
    find_threshold(hc_cnt_list, hc_text_list, ad_cnt_list, ad_text_list, best_period_threshold)


if __name__ == '__main__':
    main()
