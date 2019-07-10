from torch import nn
import torch.nn.functional as F
import sys, os
from torch import nn
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'../models'))
from RTransformer import RTransformer 

class RT(nn.Module):

    def __init__(self,  input_size, output_size, h, rnn_type, ksize, n_level, n, 
                dropout=0.2, emb_dropout=0.2, tied_weights=False):
        super(RT, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.rt = RTransformer(input_size, rnn_type, ksize, n_level, n, h, dropout)

        self.decoder = nn.Linear(input_size, output_size)
        if tied_weights:
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.rt(emb)
        y = self.decoder(y)
        return y.contiguous()

