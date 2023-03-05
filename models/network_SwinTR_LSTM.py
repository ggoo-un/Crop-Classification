# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import argparse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm

class SwinTR_Encoder(nn.Module):
    def __init__(self, out_features=1000):
        super(SwinTR_Encoder, self).__init__()

        self.model = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True, in_chans=3)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, out_features, bias=True)

    def forward(self, inputs):
        output = self.model(inputs)
        return output

class RNN_Decoder(nn.Module):
    def __init__(self, max_len, class_n, rate, embedding_dim, num_features):
        super(RNN_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features * embedding_dim, 1000)
        self.final_layer = nn.Linear(1000 + 1000, class_n)
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim = 1) # enc_out + hidden
        fc_input = concat
        output = self.dropout(self.final_layer(fc_input))
        return output

class SwinTRpth2RNN(nn.Module):
    def __init__(self, max_len, num_features, class_n, embedding_dim=512, rate=0.1):
        super(SwinTRpth2RNN, self).__init__()
        self.cnn = SwinTR_Encoder()
        self.rnn = RNN_Decoder(max_len, class_n, rate, embedding_dim, num_features)

    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)

        return output

if __name__ == '__main__':
    max_len = 144 # 24*6
    num_features = 9
    class_n = 32 # 111


    model = SwinTRpth2RNN(max_len=max_len, num_features=num_features, class_n=class_n)
    print(model)

    img = torch.randn((1, 3, 224, 224))
    seq = torch.randn((1, 9, 144))
    x = model(img, seq)
    print(x.shape)