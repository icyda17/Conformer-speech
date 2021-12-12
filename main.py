import yaml
import argparse
from utils import RandomSeed
import torch
from data.vocab import load_label_json
import json
import os
from data.dataset import MelFilterBankDataset
from torch.utils.data import DataLoader
from data.data_loader import audio_collate_fn
from models.model import Conformer
import torch.nn as nn
from utils import ScheduleAdam
from tqdm import tqdm, trange
from jiwer import wer
import numpy as np


class Manager():
    def __init__(self, mode, config_path, ckpt_name=None):

        print("Setting the configurations...")
        with open(config_path) as fp:
            self.config = yaml.load(fp, Loader=yaml.FullLoader)
            self.config = self.config['version_0']
        RandomSeed(self.config['seed_num'])

        # sub conf
        self.audio_conf = self.config['audio']
        self.device_conf = self.config['device']
        self.data_conf = self.config['data']
        self.model_conf = self.config['model']

        self.ngpus_per_node = 1
        if self.device_conf['device'] == "cuda":
            self.device_conf['device'] = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
            if self.device_conf['multi_gpu']:
                self.ngpus_per_node = torch.cuda.device_count()
        elif self.device_conf['device'] == "cpu":
            self.device_conf['device'] = torch.device('cpu')

        self.batch_size = self.config['batch_size'] * \
            self.ngpus_per_node  # 32 * 1

        print("Setting Vocab...")
        self.char2index, self.index2char = load_label_json(
            self.config['labels_path'])  # data/kor_syllable.json
        # char2index = {'_': 0, 'ⓤ': 1, '☞': 2, '☜': 3, ' ': 4, '이': 5, '다': 6, '는': 7, '에': 8 ...}
        # index2char = {0: '_', 1: 'ⓤ', 2: '☞', 3: '☜', 4: ' ', 5: '이', 6: '다', 7: '는', ...}
        self.SOS_token = self.char2index['<s>']  # 2001
        self.EOS_token = self.char2index['</s>']  # 2002
        self.PAD_token = self.char2index['_']  # 0
        self.UNK_token = self.char2index['<unk>']
        num_classes = len(self.char2index)
        print(f"Number vocab: {num_classes}")

        # Train dataset/ loader
        print("Loading data...")
        train_text_path = os.path.join(
            self.data_conf['dataset_path'], self.data_conf['train'], self.data_conf['text'])
        with open(train_text_path, 'r', encoding='utf-8') as f:
            trainData_list = json.load(f)  # data/train.json
            # print(trainData_list)
            # [{'wav': '42_0604_654_0_03223_03.wav', 'text': '자가용 끌고 가도 되나요?', 'speaker_id': '03223'}
            # ,{'wav': '41_0521_958_0_08827_06.wav', 'text': '아 네 감사합니다! 혹시 그때 4인으로 예약했는데 2명이 더 갈 거같은데 6인으로 가능한가요?', 'speaker_id': '08827'}]
        # print(len(trainData_list)) : 59662
        train_audio_path = os.path.join(
            self.data_conf['dataset_path'], self.data_conf['train'], self.data_conf['audio'])
        # print(train_dataset_path) = datasets/audio

        train_dataset = MelFilterBankDataset(audio_conf=self.config,
                                             dataset_path=train_audio_path,
                                             data_list=trainData_list,  # audio_path, text, speaker id
                                             char2index=self.char2index, sos_id=self.SOS_token, eos_id=self.EOS_token, unk_id=self.UNK_token,
                                             normalize=True,
                                             mode='train')

        self.train_loader = DataLoader(
            train_dataset, num_workers=self.config['num_workers'], batch_size=self.config['batch_size'], shuffle=True, collate_fn=audio_collate_fn)

        # Valid dataset/ loader
        valid_audio_path = os.path.join(
            self.data_conf['dataset_path'], self.data_conf['valid'], self.data_conf['audio'])
        valid_text_path = os.path.join(
            self.data_conf['dataset_path'], self.data_conf['valid'], self.data_conf['text'])

        with open(valid_text_path, 'r', encoding='utf-8') as f:
            validData_list = json.load(f)
            # print(testData_list)
            # [{"wav": "....", "text":, ...., speaker_id: ....}]

        valid_dataset = MelFilterBankDataset(audio_conf=self.config,
                                             dataset_path=valid_audio_path,
                                             data_list=validData_list,
                                             char2index=self.char2index, sos_id=self.SOS_token, eos_id=self.EOS_token, unk_id=self.UNK_token,
                                             normalize=True,
                                             mode='test')
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], collate_fn=audio_collate_fn)

        # Test dataset/ loader
        test_audio_path = os.path.join(
            self.data_conf['dataset_path'], self.data_conf['test'], self.data_conf['audio'])
        test_text_path = os.path.join(
            self.data_conf['dataset_path'], self.data_conf['test'], self.data_conf['text'])  # 'data/test.json'

        with open(test_text_path, 'r', encoding='utf-8') as f:
            testData_list = json.load(f)
            # print(testData_list)
            # [{"wav": "....", "text":, ...., speaker_id: ....}]

        test_dataset = MelFilterBankDataset(audio_conf=self.config,
                                            dataset_path=test_audio_path,
                                            data_list=testData_list,
                                            char2index=self.char2index, sos_id=self.SOS_token, eos_id=self.EOS_token, unk_id=self.UNK_token,
                                            normalize=True,
                                            mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], collate_fn=audio_collate_fn)
        print("Num of train: {}, num of valid: {}, num of test: {}".format(
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        # Model
        print("Loading the model and optimizer...")
        self.model = Conformer(
            n_classes=len(self.char2index),
            input_dim=self.model_conf['input_dim'],
            encoder_dim=self.model_conf['encoder_dim'],
            decoder_dim=self.model_conf['decoder_dim'],
            num_encoder_layers=self.model_conf['num_encoder_layers'],
            num_decoder_layers=self.model_conf['num_decoder_layers'],
            num_attention_heads=self.model_conf['num_attention_heads'],
            feed_forward_expansion_factor=self.model_conf['feed_forward_expansion_factor'],
            conv_expansion_factor=self.model_conf['conv_expansion_factor'],
            input_dropout_p=self.model_conf['input_dropout_p'],
            feed_forward_dropout_p=self.model_conf['feed_forward_dropout_p'],
            attention_dropout_p=self.model_conf['attention_dropout_p'],
            conv_dropout_p=self.model_conf['conv_dropout_p'],
            decoder_dropout_p=self.model_conf['decoder_dropout_p'],
            conv_kernel_size=self.model_conf['conv_kernel_size'],
            half_step_residual=self.model_conf['half_step_residual'],
            decoder_rnn_type=self.model_conf['decoder_rnn_type'])
        # print("[Model]")

        if self.device_conf['multi_gpu']:
            print("DataParallel...")
            self.model = nn.DataParallel(self.model).to(
                device=self.device_conf['device'])
        else:
            self.model = self.model.to(device=self.device_conf['device'])
        # Optimizer / Criterion
        self.optimizer = ScheduleAdam(
            torch.optim.Adam(self.model.parameters(),
                             betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=self.model_conf['encoder_dim'],
            warm_steps=self.model_conf['warm_steps']
        )

        self.criterion = nn.CrossEntropyLoss(
            reduction='mean').to(self.device_conf['device'])
        # Test
        if mode != "train":
            test_loss, test_cer, transcripts_list = self.evaluate(
                self.test_loader, args, save_output=True)
            # TODO: write transcript to file?
            # for line in transcripts_list:
            #     print(line)
            print("Test CER : {}".format(test_cer))

        # Train
        else:
            self.best_wer = 1e10
            self.best_loss = 1e10
            self.global_step = 0
            self.epochs_trained = 0
            self.steps_trained_in_current_epoch = 0
            if not os.path.exists(self.model_conf['ckpt_dir']):
                os.mkdir(self.model_conf['ckpt_dir'])

            if ckpt_name is not None:
                assert os.path.exists(
                    f"{self.model_conf['ckpt_dir']}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

                print("Loading checkpoint...")
                checkpoint = torch.load(
                    f"{self.model_conf['ckpt_dir']}/{ckpt_name}")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])
                self.best_wer = checkpoint['wer']
                self.global_step = checkpoint['global_step']
                self.best_loss = checkpoint['loss']
                # Check if continuing training from a checkpoint
                self.epochs_trained = self.global_step // len(
                    self.train_loader)
                self.steps_trained_in_current_epoch = self.global_step % len(
                    self.train_loader)
            else:
                print("Initializing the model...")

            print("Setting finished.")

    def train(self):
        print("Training starts.")
        self.model.zero_grad()
        no_incre_dev = 0     # for early stop
        self.max_steps = int(self.model_conf['num_epochs'])
        train_iterator = trange(
            self.epochs_trained, int(self.config['num_epochs']), desc="Epoch"
        )
        for i, _ in enumerate(train_iterator):

            print(f"#################### Epoch: {i} ####################")
            total_loss = []
            train_wer = []
            epoch_iterator = tqdm(self.train_loader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if self.steps_trained_in_current_epoch > 0:
                    self.steps_trained_in_current_epoch -= 1
                    continue
                self.model.train()

                feats, scripts, feat_lengths, script_lengths = batch
                # seqs, targets, seq_lengths, target_lengths
                # print("seqs: ", feats.size())
                # print("targets: ", scripts.size())
                # print("seq_lengths: ", feat_lengths.size())
                # print("target_lengths: ", script_lengths)

                feats, scripts, feat_lengths = feats.to(self.model_conf['device']), scripts.to(
                    self.model_conf['device']), feat_lengths.to(self.model_conf['device'])
                target = scripts[:, 1:]
                # print("target: ", target.size())

                logit = self.model(feats, feat_lengths,
                                   scripts, script_lengths)
                # print("logit: ", logit.size())
                # print("logit2: ", logit.contiguous().view(-1, logit.size(-1)).size())
                y_hat = logit.max(-1)[1]

                loss = self.criterion(
                    logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
                total_loss.append(loss.item())
                wer_score = self.get_distance(target, y_hat)
                train_wer.append(wer_score)

                loss.backward()
                epoch_iterator.set_description('Loss: {}| Wer: {}'.format(
                    round(loss.item(), 6), round(wer_score, 6)))

                if self.model_conf['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.model_conf['max_grad_norm'])
                self.optim.step()
                self.scheduler.step()
                self.optim.zero_grad()
                self.model.zero_grad()
                self.global_step += 1

                if self.max_steps > 0 and self.global_step > self.max_steps:
                    epoch_iterator.close()
                    break
            if self.max_steps > 0 and self.global_step > self.max_steps:
                train_iterator.close()
                break
            aver_loss = np.mean(total_loss)
            aver_wer = np.mean(train_wer)

            print(f"Train loss: {aver_loss} || Train wer: {aver_wer}")

            valid_loss, valid_wer = self.validation()

            if valid_loss > self.best_loss:
                self.best_loss = valid_loss
                if valid_wer <= self.best_wer:

                    state_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'global_step': self.global_step,
                        'wer': self.best_wer,
                        'loss': self.best_loss,

                    }

                    torch.save(
                        state_dict, f"{self.config['ckpt_dir']}/best_ckpt.tar")
                    print(f"***** Current best checkpoint is saved. *****")
            else:
                no_incre_dev += 1
            self.model.zero_grad()

            print(f"Best validtion wer: {self.best_wer}")
            print(
                f"Validation loss: {valid_loss} || Validation wer: {valid_wer}")
            if no_incre_dev >= self.config['max_no_incre']:
                print(
                    "early stop because there are %d epochs not increasing f1 on dev" % no_incre_dev)
                break
        print("Training finished!")

    def validation(self):
        print("Validation processing...")
        self.model.eval()

        valid_losses = []
        valid_wer = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                feats, scripts, feat_lengths, script_lengths = batch

                feats, scripts, feat_lengths = feats.to(self.model_conf['device']), scripts.to(
                    self.model_conf['device']), feat_lengths.to(self.model_conf['device'])
                target = scripts[:, 1:]

                logit = self.model(feats, feat_lengths,
                                   scripts, script_lengths)
                y_hat = logit.max(-1)[1]
                loss = self.criterion(
                    logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
                valid_losses.append(loss.item())
                wer_score = self.get_distance(target, y_hat)
                valid_wer.append(wer_score)

            aver_loss = np.mean(valid_losses)
            aver_wer = np.mean(valid_wer)

        return aver_loss, aver_wer

    def test(self):
        print("Testing starts.")
        self.model.eval()

        test_losses = []
        test_wer = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                feats, scripts, feat_lengths, script_lengths = batch

                feats, scripts, feat_lengths = feats.to(self.model_conf['device']), scripts.to(
                    self.model_conf['device']), feat_lengths.to(self.model_conf['device'])
                target = scripts[:, 1:]

                logit = self.model(feats, feat_lengths,
                                   scripts, script_lengths)
                y_hat = logit.max(-1)[1]
                loss = self.criterion(
                    logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
                test_losses.append(loss.item())
                wer_score = self.get_distance(target, y_hat)
                test_wer.append(wer_score)

            aver_loss = np.mean(test_losses)
            aver_wer = np.mean(test_wer)
        print("#################### Test results ####################")
        print(f"Test loss: {aver_loss} || Test wer: {aver_wer}")

    def get_distance(self, ref_labels, hyp_labels):
        refs = self.label_to_string(ref_labels)
        hyps = self.label_to_string(hyp_labels)

        wer_score = wer(refs, hyps)

        return wer_score

    def label_to_string(self, labels):
        # print(" labels.shape: ", labels.shape) # [8] batchsize
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == self.EOS_token:
                    break
                sent += self.index2char[j.item()]
            sents.append(sent)
        return sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=False, type=str,
                        help="The path to configuration file.", default="configs/configs.yaml")
    parser.add_argument('--mode', required=False, type=str,
                        help="Train or test?", default="train")
    parser.add_argument('--ckpt_name', required=False,
                        type=str, help="Best checkpoint file.")
    args = parser.parse_args()
    manager = Manager(args.mode, args.config_path, ckpt_name=args.ckpt_name)
    if args.mode == 'train':
        manager.train()
    manager.test()
