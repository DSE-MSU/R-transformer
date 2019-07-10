from torch import nn
import torch.nn.functional as F
import sys,os
from torch import nn
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'../models'))
from RTransformer import RTransformer 

class RT(nn.Module):
    def __init__(self, input_size, d_model, output_size, h, rnn_type, ksize, n, n_level, dropout, emb_dropout):
        super(RT, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(d_model, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        output = self.rt(x)
        output = self.linear(output).double()
        return self.sig(output)