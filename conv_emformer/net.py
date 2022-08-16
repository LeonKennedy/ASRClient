#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2022, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@gmail.com
@file: net.py
@time: 2022/5/28 09:42
@desc:
"""
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torchaudio.pipelines as P
from torch import Tensor
from torchaudio.models import Emformer
from torchaudio.models.wav2vec2.components import ConvLayerBlock, FeatureExtractor, _get_feature_extractor


class ConvEmformer(nn.Module):
    # feature_extractor: FeatureExtractor
    # conv_layer: ConvLayerBlock

    def __init__(self, token_cnt: int, right_context_length: int):
        super(ConvEmformer, self).__init__()
        self.input_dim = 384
        self.left_context_length = 25
        self.max_memory_size = 0
        self.num_layers = 7
        m = P.WAV2VEC2_ASR_LARGE_10M
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        # extractor_conv_layer_config = [(512, 10, 6)] + [(512, 5, 4)] * 2 + [(256, 3, 2)] * 2 + [(pass_length, 2, 2)] * 2
        self.feature_extractor = _get_feature_extractor(norm_mode="group_norm",
                                                        shapes=extractor_conv_layer_config, bias=False)
        self.conv_layer = ConvLayerBlock(in_channels=512,
                                         out_channels=self.input_dim,
                                         kernel_size=4,
                                         stride=4,
                                         bias=False, layer_norm=None)
        self.emformer = Emformer(self.input_dim, 6, 512,
                                 num_layers=self.num_layers,
                                 segment_length=4,
                                 dropout=0.1,
                                 left_context_length=self.left_context_length,
                                 right_context_length=right_context_length,
                                 max_memory_size=self.max_memory_size,
                                 tanh_on_mem=False,
                                 )
        self.aux = nn.Linear(in_features=self.input_dim, out_features=token_cnt)
        self.log_softmax = nn.LogSoftmax(-1)
        self.right_context_length = right_context_length
        self.input_length, self.move_length = self.input_size()
        self.input_length = torch.tensor([self.input_length], dtype=torch.int)

    def input_size(self) -> Tuple[int, int]:
        input_length, move_length = 1, 1
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2 + [(386, 4, 4),
                                                                                                (333, 5, 4)]
        for layer in extractor_conv_layer_config:
            input_length = input_length + move_length * (layer[1] - 1)
            move_length = move_length * layer[2]
        return input_length, move_length

    def init_state(self, batch_size: int, device: Optional[torch.device]) -> List[List[torch.Tensor]]:
        out = []
        for i in range(self.num_layers):
            empty_memory = torch.zeros(self.max_memory_size, batch_size, self.input_dim, device=device)
            left_context_key = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
            left_context_val = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
            past_length = torch.zeros(1, batch_size, dtype=torch.int32, device=device)
            out.append([empty_memory, left_context_key, left_context_val, past_length])
        return out

    def train(self, mode: bool = True):
        self.training = mode
        self.conv_layer.train(mode)
        self.emformer.train(mode)
        self.aux.train(mode)
        return self

    def forward(self, input, input_length):
        x, cov_len = self.feature_extractor(input, input_length)
        x = x.transpose(1, 2)
        x, length = self.conv_layer(x, cov_len)
        x = x.transpose(1, 2)
        x, length = self.emformer(x, length)
        x = self.aux(x)
        return self.log_softmax(x), length - self.right_context_length

    def stream_forward(self, wav, state=None):
        x, length = self.feature_extractor(wav, self.input_length)
        x = x.transpose(1, 2)
        x, length = self.conv_layer(x, length)
        x = x.transpose(1, 2)
        x, _, state = self.emformer.infer(x, torch.tensor([x.size(1)], dtype=torch.int), state)
        x = self.aux(x)
        x = torch.argmax(x.squeeze(), dim=-1)
        return x, state


class LiteStreamEmformer(ConvEmformer):

    def unfold_state(self, state: List[List[Tensor]]) -> Tensor:
        return torch.cat([torch.cat([e.flatten() for e in row]) for row in state])

    def fold_state(self, state: Tensor) -> List[List[Tensor]]:
        left_context_key_length = self.input_dim * self.left_context_length
        left_context_val_length = self.input_dim * self.left_context_length
        size = left_context_key_length + left_context_val_length + 1
        out: List[List[Tensor]] = []
        for row in torch.split(state, size):
            empty_memory = torch.zeros(0, 1, self.input_dim)
            past_length = torch.zeros(1, 1, dtype=torch.int32)
            left_context_key = row[:left_context_key_length].resize(self.left_context_length, 1, self.input_dim)
            left_context_val = row[left_context_key_length:left_context_key_length + left_context_val_length].resize(
                self.left_context_length, 1, self.input_dim)
            past_length[0, 0] = row[-1]
            a: List[Tensor] = [empty_memory, left_context_key, left_context_val, past_length]
            out.append(a)
        return out

    def forward(self, wav: Tensor, state: Tensor):
        fold_state = self.fold_state(state)
        x, length = self.feature_extractor(wav, torch.tensor([wav.size(1)], dtype=torch.int))
        x = x.transpose(1, 2)
        x, length = self.conv_layer(x, length)
        x = x.transpose(1, 2)
        x, _, fold_state = self.emformer.infer(x, torch.tensor([x.size(1)], dtype=torch.int), fold_state)
        x = self.aux(x)
        x = torch.argmax(x.squeeze(), dim=-1)
        return x, self.unfold_state(fold_state)


class LiteEmformer(ConvEmformer):
    def forward(self, wav):
        x, length = self.feature_extractor(wav, torch.tensor([wav.size(1)], dtype=torch.int))
        x = x.transpose(1, 2)
        x, length = self.conv_layer(x, length)
        x = x.transpose(1, 2)
        x, length = self.emformer(x, torch.tensor([x.size(1)], dtype=torch.int))
        x = self.aux(x)
        x = torch.argmax(x.squeeze(), dim=-1)
        return x


class InferenceEmformer(ConvEmformer):
    def forward(self, wav, wav_length):
        x, length = self.feature_extractor(wav, wav_length)
        x = x.transpose(1, 2)
        x, length = self.conv_layer(x, length)
        x = x.transpose(1, 2)
        x, length = self.emformer(x, length)
        x = self.aux(x)
        x = x.argmax(-1)
        return x, length - self.right_context_length
