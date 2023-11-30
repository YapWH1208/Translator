import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 32
epoch_num = 100
save_interval = 1000
early_stop = 5
lr = 3e-4

# greed decode的最大句子长度
max_len = 60
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

data_dir = '../data'
xml_folder = '../data/xml'
train_data_path = '../data/json/train.json'
dev_data_path = '../data/json/dev.json'
test_data_path = '../data/json/test.json'
model_path = '../experiment'
log_path = '../experiment/train.log'
output_path = '../experiment/output.txt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')