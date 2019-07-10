import torch.nn.functional as F
import os, sys
from torch import nn
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'../models'))
from RTransformer import RTransformer 


class RT(nn.Module):
    def __init__(self, input_size, d_model, output_size, h, rnn_type, ksize, n_level, n, dropout=0.2, emb_dropout=0.2):
        super(RT, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = x.transpose(-2,-1)
        x = self.encoder(x)
        x = self.rt(x)  # input should have dimension (N, C, L)
        x = x.transpose(-2,-1)
        o = self.linear(x[:, :, -1])
        return F.log_softmax(o, dim=1)



