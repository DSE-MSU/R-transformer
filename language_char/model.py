from torch import nn
import torch.nn.functional as F
import sys, os
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'../models'))
from RTransformer import RTransformer 

class RT(nn.Module):
    def __init__(self, input_size, output_size, h, n, rnn_type, ksize,  n_level, dropout, emb_dropout):
        super(RT, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.rt = RTransformer(input_size, rnn_type, ksize, n_level, n, h, dropout)
        self.decoder = nn.Linear(input_size, output_size)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.rt(emb)
        o = self.decoder(y)
        return o.contiguous()