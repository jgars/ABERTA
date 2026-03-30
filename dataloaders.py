from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import core
import os
import re


class ADReSSTextDataset(Dataset):
    def get_file_text(self, file_path, level_list, punctuation_list):
        return core.load_file(file_path, level_list, punctuation_list)

    def get_audio_embedding(self, file_path):
        file_path = file_path.replace('asr_text', 'asr_embedding').replace('.txt', '.npy')
        return np.load(file_path)


class adresso21TextTrainDataset(ADReSSTextDataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, sentiment in (('cn', 0), ('ad', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = self.get_file_text(file_path, level_list, punctuation_list)
                text_file['audio_embedding'] = self.get_audio_embedding(file_path)
                if len(text_file['text'].split()) > filter_min_word_length:
                    self.X.append(text_file)
                    self.Y.append(sentiment)
                    self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                    self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        data = pd.read_csv(os.path.join(dir_path, 'adresso-train-mmse-scores.csv'))
        df = pd.DataFrame(data)

        for _, row in df.iterrows():
            labels[row['adressfname']] = int(row['mmse'])
        print(labels)
        return labels
    

class adresso21TextTestDataset(ADReSSTextDataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        # Load ground truth (Task 1)
        gt_csv = pd.read_csv(os.path.join(dir_path, 'ground_truth', 'task1.csv'))
        gt_dict = {}
        for _, row in gt_csv.iterrows():
            audio_name = row[0]
            if row[1] == "Control":
                target_label = 0
            else:
                target_label = 1
            gt_dict[audio_name] = target_label
        # Load ground truth (Task 2)
        gt2_csv = pd.read_csv(os.path.join(dir_path, 'ground_truth', 'task2.csv'))
        gt2_dict = {}
        for _, row in gt2_csv.iterrows():
            audio_name = row[0]
            gt2_dict[audio_name] = row[1]
        # Load text and audio
        folder = os.path.join(dir_path, 'asr_text')
        for name in tqdm(sorted(os.listdir(folder))):
            audio_name = name.replace(".txt", "")
            file_path = os.path.join(folder, name)
            text_file = self.get_file_text(file_path, level_list, punctuation_list)
            text_file['audio_embedding'] = self.get_audio_embedding(file_path)
            if len(text_file['text'].split()) > filter_min_word_length:
                self.X.append(text_file)
                self.Y.append(gt_dict[audio_name])
                self.Y_mmse.append(gt2_dict[audio_name])
                self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
        }

    def __len__(self):
        return len(self.X)
