import os
from torch.utils.data import Dataset
import torchaudio
from torch import Tensor
import numpy as np
from data.audio.augment import SpecAugment, NoiseInjector, TimeStretchAugment
from pathlib import Path
from data.audio.audio_utils import load_audio
import torch


class FilterBankFeatureTransform():

    def __init__(self, num_mels, window_length, window_stride):
        super(FilterBankFeatureTransform, self).__init__()
        self.num_mels = num_mels
        self.window_length = window_length
        self.window_stride = window_stride
        self.function = torchaudio.compliance.kaldi.fbank

    def __call__(self, signal):
        return self.function(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.num_mels,
            frame_length=self.window_length,
            frame_shift=self.window_stride
        ).transpose(0, 1).numpy()


class MelFilterBankDataset(Dataset):

    NONE_AUGMENT = 0
    SPEC_AUGMENT = 1
    NOISE_AUGMENT = 2
    TIME_STRETCH = 3
    AUDIO_JOINING = 4

    def __init__(self, conf, dataset_path, data_list, char2index, sos_id, eos_id, unk_id, normalize=False, mode='train'):
        """
        Dataset for audio & transcript matching
        :param audio_conf: Sample rate, window, window size length, stride
        :param data_list: dictionary . key: 'wav', 'text', 'speaker_id'
        :param char2index: character to index mapping Dictionary
        :param normalize: Normalized by instance-wise standardazation
        # TODO: 
        # del_silence (bool): flag indication whether to apply delete silence or not
        # apply_joining_augment (bool): flag indication whether to apply audio joining augment or not

        apply_spec_augment (bool): flag indication whether to apply spec augment or not
        apply_noise_augment (bool): flag indication whether to apply noise augment or not
        apply_time_stretch_augment (bool): flag indication whether to apply time stretch augment or not

    """

        super(MelFilterBankDataset, self).__init__()
        # dict{sample rate, window_size, window_stride}
        self.mode = mode
        self.audio_conf = conf['audio']
        self.augment_conf = conf['spec_augment']
        self.data_list = data_list  # [{"wav": , "text": , "speaker_id": "}]
        self.audio_paths = []
        self.transcripts = []
        self.size = len(self.data_list)  # 59662
        self.char2index = char2index
        self.sos_id = sos_id  # 2001
        self.eos_id = eos_id  # 2002
        self.unk_id = unk_id
        self.PAD = 0
        self.normalize = normalize  # Train: True
        self.augments = [self.NONE_AUGMENT] * len(self.audio_paths)
        self.dataset_path = dataset_path  # datasets/audio
        self.transforms = FilterBankFeatureTransform(
            self.audio_conf["num_mel"], self.audio_conf["window_size"], self.audio_conf["window_stride"]
        )
        self.apply_spec_augment = self.augment_conf['apply_spec_augment']
        self.apply_noise_augment = self.augment_conf['apply_noise_augment']
        self.apply_time_stretch_augment = self.augment_conf['apply_time_stretch_augment']

        if self.mode != 'train' or not any([self.apply_noise_augment, self.apply_spec_augment, self.apply_time_stretch_augment]):
            for idx in range(self.size):
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.augments.append(self.NONE_AUGMENT)
        else:
            if self.apply_spec_augment:
                self._spec_augment = SpecAugment(
                    freq_mask_para=self.augment_conf['freq_mask_para'],
                    freq_mask_num=self.augment_conf['freq_mask_num'],
                    time_mask_num=self.augment_conf['time_mask_num'],
                )
                for idx in range(self.size):
                    self.audio_paths.append(self.audio_paths[idx])
                    self.transcripts.append(self.transcripts[idx])
                    self.augments.append(self.SPEC_AUGMENT)

            if self.apply_noise_augment:
                if Path(self.augment_conf['noise_dataset_dir']).is_dir():
                    raise ValueError("Directory doesn't exist %s" %
                                     self.augment_conf['noise_dataset_dir'])

                self._noise_injector = NoiseInjector(
                    noise_dataset_dir=self.augment_conf['noise_dataset_dir'],
                    sample_rate=self.augment_conf['noise_sample_rate'],
                    noise_level=self.augment_conf['noise_level'],
                )
                for idx in range(self.size):
                    self.audio_paths.append(self.audio_paths[idx])
                    self.transcripts.append(self.transcripts[idx])
                    self.augments.append(self.NONE_AUGMENT)

            if self.apply_time_stretch_augment:
                self._time_stretch_augment = TimeStretchAugment(
                    min_rate=self.augment_conf['time_stretch_min_rate'],
                    max_rate=self.augment_conf['time_stretch_max_rate'],
                )
                for idx in range(self.size):
                    self.audio_paths.append(self.audio_paths[idx])
                    self.transcripts.append(self.transcripts[idx])
                    self.augments.append(self.TIME_STRETCH)

        self.total_size = len(self.audio_paths)

    def __getitem__(self, index):
        wav_name = self.audio_paths[index]
        # print("wav: " , wav_name) # 41_0607_213_1_08139_05.wav
        audio_path = os.path.join(self.dataset_path, wav_name)
        # print("audio_path: ", audio_path): data/wavs_train/41_0607_213_1_08139_05.wav
        transcript = self.transcripts[index]
        # print("text: ", transcript): 예약 받나요?

        spect = self._parse_audio(audio_path, self.augments[index])
        # print("spect: ", spect.size()) #
        transcript = self.parse_transcript(transcript)
        # print("text: ", transcript) #
        return spect, transcript

    def _parse_audio(self, audio_path, augment):
        signal = load_audio(
            audio_path, sample_rate=self.audio_conf['sample_rate'])

        if augment == self.TIME_STRETCH:
            signal = self._time_stretch_augment(signal)

        if augment == self.NOISE_AUGMENT:
            signal = self._noise_injector(signal)

        feature = self.transforms(signal)

        # normalize
        feature -= feature.mean()
        feature /= np.std(feature)

        feature = torch.FloatTensor(feature).transpose(0, 1)

        if augment == self.SPEC_AUGMENT:
            feature = self._spec_augment(feature)

        return feature

    def parse_transcript(self, transcript):
        # print(list(transcript))
        # ['미', '리', ' ', '예', '약', '하', '려', '고', ' ', '하', '는', '데', '요', '.']

        transcript = [self.char2index.get(x, self.unk_id)
                      for x in list(transcript)]
        # filter(조건, 순횐 가능한 데이터): char2index 의 key 에 없는 것(None) 다 삭제 해버림
        # print("transcript: ", transcript):[49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126]

        transcript = [self.sos_id] + transcript + [self.eos_id]
        # [2001, 49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126, 2002]

        return transcript

    def __len__(self):
        return self.total_size  # 59662
