import yaml
import argparse
from utils import RandomSeed
import torch
from data.vocab import load_label_json
import json
import os
from data.dataset import MelFilterBankDataset

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
        num_classes = len(self.char2index)   
        print(f"Number vocab: {num_classes}")    

        # Train dataset/ loader
        with open(self.data_conf['train_file'], 'r', encoding='utf-8') as f:
            trainData_list = json.load(f) # data/train.json
            # print(trainData_list)
            # [{'wav': '42_0604_654_0_03223_03.wav', 'text': '자가용 끌고 가도 되나요?', 'speaker_id': '03223'}
            # ,{'wav': '41_0521_958_0_08827_06.wav', 'text': '아 네 감사합니다! 혹시 그때 4인으로 예약했는데 2명이 더 갈 거같은데 6인으로 가능한가요?', 'speaker_id': '08827'}]
        # print(len(trainData_list)) : 59662        
        train_dataset_path = os.path.join(self.data_conf['dataset_path'], self.data_conf['audio_dir'])
        # print(train_dataset_path) = datasets/audio  

        
        train_dataset = MelFilterBankDataset(audio_conf=self.audio_conf,
                                        dataset_path=train_dataset_path,
                                        data_list=trainData_list, # audio_path, text, speaker id
                                        char2index=self.char2index, sos_id=self.SOS_token, eos_id=self.EOS_token,
                                        normalize=True,
                                        mode='train')

        train_sampler = BucketingSampler(data_source=train_dataset, batch_size=batch_size)
        train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)

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