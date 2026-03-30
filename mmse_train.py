import os
import time
import json
from datetime import datetime

import transformers
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import dataloaders
from helpers import get_device
from layers import TransformerAttentionPooling, CustomTransformerEncoderLayer

class WERT(nn.Module):
    def __init__(self, model_name):
        super(WERT, self).__init__()

        # Get the default BERT config
        self.text_conf = transformers.BertConfig.from_pretrained(model_name)
        self.text_conf.train_original = True
        self.text_conf.word_predictor_pre_training = False

        # BERT tokenizer and model
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name, model_max_length=512)
        self.text_model = transformers.BertModel.from_pretrained(model_name, config=self.text_conf)
        
        embedding_size = 768

        self.combined_attention = CustomTransformerEncoderLayer(d_model=embedding_size, nhead=8, batch_first=True)
        self.linear_output = nn.Linear(embedding_size, 1)
        self.text_ln = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=0.2)
        self.extract_audio = TransformerAttentionPooling(1024, self.text_conf.hidden_size, use_posenc=True)
        self.final_pool = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 768))
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)
        self.out_ln = nn.LayerNorm(768)
        

    def forward(self, text_features, audio_features, audio_pad_mask):
        # Text pipeline
        text_outputs = self.text_model(**text_features)
        text_outputs = text_outputs[0]

        text_outputs = self.text_ln(text_outputs)
        text_outputs = self.dropout(text_outputs)

        # Create windows to group the audio embeddings
        n_text = text_outputs.shape[1]
        n_audio = audio_features.shape[1]

        # Calculate a window size that allows to group the audio embeddings and get the same number of audio embedding that text embeddings
        window_size = (n_audio + n_text - 1) // n_text
        num_embeddings_needed = n_text * window_size

        # Get the indixes needed to group the audio embeddings
        k = torch.arange(num_embeddings_needed, device="cuda")

        # Multiply the number of existing audio embeddings with the number of embeddings needed, then divide by number of embeddings needed.
        # As we don't have enough audio embeddings, when we divide and round some indices will be repeated, that repeated indixes will be 
        # the embeddings that will be used to fill the windows and get groups of the same size
        idx = torch.div(k * n_audio, num_embeddings_needed, rounding_mode="floor")

        # Repeat the idx for every batch (expand is like repeated but optimized to use with dimensions of size 1) 
        idx_flat = idx.reshape(1, -1).expand(audio_features.shape[0], -1) 

        # We expand again to set the indices of the size of the audio embeddings
        gather_idx = idx_flat.unsqueeze(-1).expand(-1, -1, audio_features.shape[2])

        # audio_pad_mask: (B, T_audio) -> True=PAD
        pad_gather_idx = idx_flat

        # Gather masks in dim=1
        audio_pad_mask_flat = torch.gather(audio_pad_mask, dim=1, index=pad_gather_idx)  # (B, n_text*window_size)
        audio_pad_mask_win = audio_pad_mask_flat.view(audio_features.shape[0], n_text, window_size)  # (B, n_text, window_size)
        audio_grouped_flat = torch.gather(audio_features, dim=1, index=gather_idx)

        # Create a view with the size of the window
        audio_outputs = audio_grouped_flat.view(audio_features.shape[0], n_text, window_size, audio_features.shape[2])

        # Extract the features of the windows
        audio_outputs, attn_audio = self.extract_audio(audio_outputs, key_padding_mask=audio_pad_mask_win)
        B, n_text, _ = text_outputs.shape

        # Text: Attention_mask==0 -> PAD
        text_pad_mask = (text_features["attention_mask"] == 0) # (B, n_text) bool

        # Audio: One audio token is PAD is all the window was PAD
        audio_token_pad_mask = audio_pad_mask_win.all(dim=2) # (B, n_text) bool

        # Combined data: [text, audio]
        output = torch.cat([text_outputs, audio_outputs], dim=1) # (B, 2*n_text, 768)
        combined_pad_mask = torch.cat([text_pad_mask, audio_token_pad_mask], dim=1)  # (B, 2*n_text)

        # Multimodal self-attention
        output, attn_comb = self.combined_attention(output, src_key_padding_mask=combined_pad_mask)

        # Final attention pooling
        q = self.pool_query.expand(B, 1, output.size(-1))  # (B,1,768)
        pooled, attn_pool = self.final_pool(
            q, output, output,
            key_padding_mask=combined_pad_mask 
        )
        pooled = pooled.squeeze(1)  # (B,768)
        pooled = self.dropout(self.out_ln(pooled))
        pred = self.linear_output(pooled) # (B,1)
        pred = pred.squeeze(-1) # (B,)

        # attentions
        attn_dict = {"audio": attn_audio, "combined": attn_comb, "pool": attn_pool}

        return pred, attn_dict



def train_model(model, args, use_kfold = False):
    # Prepare the model, optimizer and loss
    device = get_device()
    model.to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=args['learning_rate'])
    criterion = nn.MSELoss()

    # Load datasets
    train_dataset = dataloaders.adresso21TextTrainDataset(
        'data/adresso21/diagnosis/train', args['level_list'], args['punctuation_list'],
        filter_min_word_length=args['train_filter_min_word_length'])
    test_dataset = dataloaders.adresso21TextTestDataset(
        'data/adresso21/diagnosis/test-dist', args['level_list'], args['punctuation_list'],
        filter_min_word_length=args['test_filter_min_word_length'])
    
    def collate_fn(data_list):
        inputs, labels = [], []
        mmse_label = []
        file_idx_list = []
        audio_embeddings = []
        for data in data_list:
            inputs.append(data['text'])
            labels.append(data['label'])
            mmse_label.append(data['label_mmse'])
            file_idx_list.append(data['file_idx'])
            audio_embeddings.append(data['audio_embedding'])

        inputs = model.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return {
            'file_idx': file_idx_list,
            'inputs': inputs,
            'label': torch.tensor(labels),
            'label_mmse': torch.FloatTensor(mmse_label),
            'audio_embeddings': audio_embeddings,
        }
    
    if not use_kfold:
        # Create the dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'],
                                                shuffle=True, num_workers=4, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                shuffle=False, num_workers=4, collate_fn=collate_fn)
        num_epochs = 400
        report_dict_train = []
        report_dict_test = []
        best_model_metric = float("inf")
        for epoch in range(num_epochs):
            # Train loader
            epoch_dict_train = run_epoch(model, optimizer, criterion, train_loader, "train", epoch, num_epochs)
            report_dict_train.append(epoch_dict_train)
            # Test loader
            epoch_dict_test = run_epoch(model, optimizer, criterion, test_loader, "test", epoch, num_epochs)
            report_dict_test.append(epoch_dict_test)
            # Save best model
            test_metric = epoch_dict_test["rmse"]
            if test_metric < best_model_metric:
                # Save model and update best model metric
                torch.save(model.state_dict(), args['log_model_path'] + "/best_model.pt")
                best_model_metric = test_metric
                print(f"Saving new best model with RMSE {best_model_metric:.4f}")
                # Save metrics
                with open(args['log_dir'] + "training_logs/best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(epoch_dict_test, f, ensure_ascii=False, indent=4)
    else:
        pass


def run_epoch(model, optimizer, criterion, dataloader, phase, epoch, num_epochs):
    # Show info 
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    # Set model to training mode
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    pred_list = []
    target_list = []

    device = get_device()
    for batch_data in dataloader:
        # Clears existing gradients from previous epoch
        optimizer.zero_grad()

        # Get inputs
        text_inputs = batch_data['inputs']
        audio_inputs = []
        for audio_embedding in batch_data['audio_embeddings']:
            audio_embedding = torch.FloatTensor(audio_embedding)
            audio_inputs.append(audio_embedding)
        audio_inputs = pad_sequence(audio_inputs, batch_first=True).to(device)

        lengths = torch.tensor([a.shape[0] for a in batch_data['audio_embeddings']], device=device)
        max_len = audio_inputs.size(1)
        audio_pad_mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]  # (B, T_audio), True=PAD

        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(device)
        target = batch_data['label_mmse'].to(device)

        with torch.set_grad_enabled(phase == 'train'):
            pred, _ = model(text_inputs, audio_inputs, audio_pad_mask)   # pred: (B,)
            loss = criterion(pred, target)  # MSE

            # Backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # Statistics
        running_loss += loss.item() * target.size(0)
        pred_list.extend(pred.detach().cpu().tolist())
        target_list.extend(target.detach().cpu().tolist())

    pred_arr = np.array(pred_list, dtype=np.float32)
    target_arr = np.array(target_list, dtype=np.float32)
    
    pred_int = np.rint(pred_arr)
    pred_int = np.clip(pred_int, 0, 30)

    mse = np.mean((pred_int - target_arr) ** 2)
    rmse = float(np.sqrt(mse))

    epoch_loss = running_loss / len(dataloader.dataset)

    print(f"{phase} mse_loss: {epoch_loss:.8f} rmse: {rmse:.4f}")

    report_dict = {
        "loss_mse": float(epoch_loss),
        "rmse": rmse,
    }
    return report_dict

    
def main():
    # Use an empty dict to insert the args
    args = {}

    # Create a different folder for each run
    args['model_description'] = 'bert_base_sequence_level_2-83_123'

    # Other args
    args['learning_rate'] = 2e-5
    args['level_list'] = (83, 123,)
    args['punctuation_list'] = (',', '.',)
    args['train_filter_min_word_length'] = 20
    args['test_filter_min_word_length'] = 0
    args['batch_size'] = 16
    args['model_name'] = "bert-base-uncased"
    args['log_dir'] = os.path.join(os.path.abspath("log"), args['model_description'])
    args['log_dir'] += f"/{datetime.now()}/"
    os.makedirs(args['log_dir'])

    model_dir_path = os.path.join(args['log_dir'], 'models')
    os.makedirs(model_dir_path)
    args['log_model_path'] = model_dir_path

    training_log_dir_path = os.path.join(args['log_dir'], 'training_logs')
    os.makedirs(training_log_dir_path)
    
    # Get model and device
    model = WERT(args['model_name'])

    # Start training
    train_model(model, args)



if __name__ == '__main__':
    main()
