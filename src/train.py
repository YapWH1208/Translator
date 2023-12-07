import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import chinese_tokenizer_load, lastest_checkpoint


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, model, criterion, optimizer):
    """训练并保存模型"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model, LossCompute(model.generator, criterion, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        model.eval()
        dev_loss = run_epoch(dev_data, model, LossCompute(model.generator, criterion))
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path+"/model_{}.pth".format(epoch))
            best_bleu_score = bleu_score
            early_stop = config.early_stop
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


def evaluate(data, model, mode='dev', use_beam=True):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w", encoding='utf-8') as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(lastest_checkpoint()))
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model, LossCompute(model.generator, criterion))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    with torch.no_grad():
        last_model = lastest_checkpoint()
        model.load_state_dict(torch.load(last_model))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
        print(translation[0])
