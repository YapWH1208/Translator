import torch
import json
import numpy as np
import os, re
import random
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

import config
DEVICE = config.device

def build_dataset(xml_folder, train_data_path, dev_data_path, test_data_path, max_length, prob=0.85):
    data = []
    
    # Read data from XML files
    for file in os.listdir(xml_folder):
        if file.endswith('.xml'):
            with open(f'{xml_folder}/{file}', 'r', encoding='utf-8') as f:
                content = f.read()
                en_lines = re.findall(r'<en>(.*?)</en>', content)
                zh_lines = re.findall(r'<zh>(.*?)</zh>', content)
                
                i = 0
                while i < len(en_lines):
                    if zh_lines[i] == '':
                        i += 1
                        continue
                    en_line = en_lines[i].replace('\\n', '\n')
                    zh_line = zh_lines[i].replace('\\n', '\n')
                    
                    if len(en_line) > max_length:
                        i += 1
                        continue
                        
                    while random.random() < prob and i+1 < len(en_lines) and zh_lines[i+1] != '' and len(en_line)+len(en_lines[i+1])+1 <= max_length:
                        i += 1
                        en_line += '\n' + en_lines[i].replace('\\n', '\n')
                        zh_line += '\n' + zh_lines[i].replace('\\n', '\n')
                    
                    data.append([en_line, zh_line])
                    i += 1

    # Shuffle the data
    random.shuffle(data)

    # Split data into train, dev, and test sets
    test_size = int(0.1 * len(data))
    dev_size = int(0.1 * len(data))
    train_data = data[:-(dev_size + test_size)]
    dev_data = data[-(dev_size + test_size):-test_size]
    test_data = data[-test_size:]

    # Write data to files
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)

    with open(dev_data_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False)

    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # Set the size of the subsequent_mask
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        # If trg is not None, then masked.
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder input
            self.trg = trg[:, :-1]
            # true label
            self.trg_y = trg[:, 1:]
            # Mask out the decoder input
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask operation
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """Sort the dataset according to the length of the English sentence"""
        dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        out_en_sent = []
        out_cn_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_cn_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)

if __name__ == '__main__':
    build_dataset(config.xml_folder, config.train_data_path, config.dev_data_path, config.test_data_path, config.max_len)