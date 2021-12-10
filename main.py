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

        self.batch_size = self.config['batch_size'] * self.ngpus_per_node # 32 * 1

        print("Setting Vocab...")
        self.char2index, self.index2char = load_label_json(self.config['labels_path']) # data/kor_syllable.json
        # char2index = {'_': 0, 'ⓤ': 1, '☞': 2, '☜': 3, ' ': 4, '이': 5, '다': 6, '는': 7, '에': 8 ...}
        # index2char = {0: '_', 1: 'ⓤ', 2: '☞', 3: '☜', 4: ' ', 5: '이', 6: '다', 7: '는', ...}
        self.SOS_token = self.char2index['<s>'] # 2001
        self.EOS_token = self.char2index['</s>'] # 2002
        self.PAD_token = self.char2index['_'] # 0
        self.UNK_token = self.char2index['<unk>']
        num_classes = len(self.char2index)   
        print(f"Number vocab: {num_classes}")    

        # Train dataset/ loader
        print("Loading data...")
        train_text_path = os.path.join(self.data_conf['dataset_path'], self.data_conf['train'], self.data_conf['text'])
        with open(train_text_path, 'r', encoding='utf-8') as f:
            trainData_list = json.load(f) # data/train.json
            # print(trainData_list)
            # [{'wav': '42_0604_654_0_03223_03.wav', 'text': '자가용 끌고 가도 되나요?', 'speaker_id': '03223'}
            # ,{'wav': '41_0521_958_0_08827_06.wav', 'text': '아 네 감사합니다! 혹시 그때 4인으로 예약했는데 2명이 더 갈 거같은데 6인으로 가능한가요?', 'speaker_id': '08827'}]
        # print(len(trainData_list)) : 59662        
        train_audio_path = os.path.join(self.data_conf['dataset_path'], self.data_conf['train'], self.data_conf['audio'])
        # print(train_dataset_path) = datasets/audio  

        
        train_dataset = MelFilterBankDataset(audio_conf=self.config,
                                        dataset_path=train_audio_path,
                                        data_list=trainData_list, # audio_path, text, speaker id
                                        char2index=self.char2index, sos_id=self.SOS_token, eos_id=self.EOS_token, unk_id=self.UNK_token,
                                        normalize=True,
                                        mode='train')

        self.train_loader = DataLoader(train_dataset, num_workers=self.config['num_workers'], batch_size=self.config['batch_size'], shuffle=True, collate_fn=audio_collate_fn)


        # Test dataset/ loader
        test_audio_path = os.path.join(self.data_conf['dataset_path'], self.data_conf['test'], self.data_conf['audio'])
        test_text_path = os.path.join(self.data_conf['dataset_path'], self.data_conf['test'], self.data_conf['text']) # 'data/test.json'

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
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], collate_fn=audio_collate_fn)
        print("Num of train: {}, num of test: {}".format(len(train_dataset), len(test_dataset)))
        # Model
        print("Loading the model and optimizer...")
        self.model = Conformer(
            n_classes=len(self.char2index),
            input_dim=self.model_conf['input_dim'],
            encoder_dim= self.model_conf['encoder_dim'],
            decoder_dim= self.model_conf['decoder_dim'],
            num_encoder_layers= self.model_conf['num_encoder_layers'],
            num_decoder_layers= self.model_conf['num_decoder_layers'],
            num_attention_heads= self.model_conf['num_attention_heads'],
            feed_forward_expansion_factor= self.model_conf['feed_forward_expansion_factor'],
            conv_expansion_factor= self.model_conf['conv_expansion_factor'],
            input_dropout_p= self.model_conf['input_dropout_p'],
            feed_forward_dropout_p=self.model_conf['feed_forward_dropout_p'],
            attention_dropout_p=self.model_conf['attention_dropout_p'],
            conv_dropout_p= self.model_conf['conv_dropout_p'],
            decoder_dropout_p= self.model_conf['decoder_dropout_p'],
            conv_kernel_size= self.model_conf['conv_kernel_size'],
            half_step_residual= self.model_conf['half_step_residual'],
            decoder_rnn_type= self.model_conf['decoder_rnn_type'])
        # print("[Model]")

        if self.device_conf['multi_gpu']:
            print("DataParallel...")
            self.model = nn.DataParallel(self.model).to(device=self.device_conf['device'])
        else:
            self.model = self.model.to(device=self.device_conf['device'])
        # Optimizer / Criterion
        self.optimizer = ScheduleAdam(
            torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=self.model_conf['encoder_dim'],
            warm_steps=self.model_conf['warm_steps']
        )

        criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device_conf['device'])
        # Test
        if args.mode != "train":
            # TODO: Continue https://github.com/park-cheol/ASR-Conformer/blob/master/main.py https://github.com/icyda17/bert-crf/blob/main/main.py
            test_loss, test_cer, transcripts_list = self.evaluate(self.test_loader, criterion, args, save_output=True)

            # for line in transcripts_list:
            #     print(line)
            print("Test CER : {}".format(test_cer))

        # Train
        else:
            best_cer = 1e10

            for epoch in range(args.start_epoch, args.epochs):
                train_loss, train_cer = train(model, train_loader, criterion, optimizer, args, epoch, train_sampler,
                                            args.max_norm)
                # args.max_norm = 400

                cer_list = []
                for test_file in args.test_file_list:
                    print(test_file)
                    test_loader = testLoader_dict[test_file] # test.json
                    test_loss, test_cer, _ = evaluate(model, test_loader, criterion, args, save_output=False)
                    test_log = 'Test({name}) Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\tAverage CER {cer:.3f}\t'.format(
                        epoch + 1, name=test_file, loss=test_loss, cer=test_cer)
                    print(test_log)

                    cer_list.append(test_cer)

                if best_cer > cer_list[0]:
                    print("Found better validated model")
                    torch.save(model.state_dict(), "saved_models/model_%d.pth" % (epoch + 1))
                    best_cer = cer_list[0]

                print("Shuffling batches...")
                train_sampler.shuffle(epoch)   

    def evaluate(self):
        pass     
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