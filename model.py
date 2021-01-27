#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from utils import homeDirectory,srcIndex, mu, sig

SEED = 3456

# init pytorch and random seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# define device to cuda if cuda is available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define Encoder(
# input_dim,
# emb_dim,
# hid_dim,
# n_layers,
# dropout
# )

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder,self).__init__()        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim, emb_dim)
        if n_layers == 1:
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
        else:
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        #src = [src len, batch size, feature]
        s,b,h = src.size()
        #change the view to feed it into the embedding layer
        x = src.view(s*b,h)
        x = self.embedding(x)
        #change the view to feed is into the lstm layer
        x = x.view(s,b,-1)
        embedded = self.dropout(x)
        #embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
	#outputs =  [src len,  batch size, hid dim]
        #hidden  =  [n layers, batch size, hid dim]
        #cell    =  [n layers, batch size, hid dim]        
        #outputs are always from the top hidden layer
        return hidden, cell

# define Decoder(
# output_dim,
# emb_dim,
# hid_dim,
# n_layers,
# dropout
#)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(output_dim, emb_dim)
        if n_layers == 1:
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
        else:
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        #input = [batch size, feature]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        embedded = self.dropout(self.embedding(input))
        embedded = embedded.unsqueeze(0)
        #embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        prediction = self.out(output.squeeze(0))
        #prediction = [batch size, output dim]
        return prediction, hidden, cell

# define Seq2Seq(
# encoder,
# decoder,
# device
#)
# the hid_dim and n_layers in encoder and decoder should be the same

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size, src feature]
        #trg = [trg len, batch size, trg feature]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_feature = trg.shape[2]
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_feature).to(self.device)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(0, max_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            if t < max_len -1:
                input = trg[t+1] if teacher_force else output
        return outputs
INPUT_DIM = 5
OUTPUT_DIM = 2
ENC_EMB_DIM = 24
DEC_EMB_DIM = 24
HID_DIM = 32
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

class Model:
    def __init__(self):
        INPUT_DIM = 6
        OUTPUT_DIM = 2
        ENC_EMB_DIM = 24
        DEC_EMB_DIM = 24
        HID_DIM = 32
        N_LAYERS = 2
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
        self.model = Seq2Seq(enc, dec, device).to(device)
        torch.cuda.FloatTensor(2)
        self.model.load_state_dict(torch.load(homeDirectory+'my-model-test.pt'))
        self.__holdplace = torch.zeros(8, 1, 2).to(device)

    def predictFuture(self, npLocations):
        # 1. 转换为 numpy 形式
        # npLocations = np.array(locations)  # locations 的维度是 timestep * 6
        centers = np.copy(npLocations[-1, 0:2])
        # 2. 去中心化，以最后一个点为中心
        npLocations[:,0:2] = (npLocations[:,0:2] - centers[np.newaxis, :] - mu[np.newaxis, :])/sig[np.newaxis, :]
        npLocations[:,2:4] = (npLocations[:,2:4] - centers[np.newaxis, :] - mu[np.newaxis, :])/sig[np.newaxis, :]
        npLocations[:,4] = (npLocations[:,4] - centers[np.newaxis, 0] - mu[np.newaxis, 0])/sig[np.newaxis, 0]
        npLocations[:,5] =  npLocations[:,5] / 540
        npLocations = npLocations.reshape(11, 1, 6)
        # 3. 轨迹预测
        tsLocations = torch.tensor(npLocations, dtype=torch.float32).to(device)
        self.model.eval()
        self.__holdplace[0][0][0] = tsLocations[10][0][0]
        self.__holdplace[0][0][1] = tsLocations[10][0][1]
        tsFuture = self.model(tsLocations[:10], self.__holdplace, 0)
        npFuture = tsFuture.cpu().detach().numpy().reshape(8, 2)
        # 4. 恢复原来的中心
        npFuture = npFuture * sig[np.newaxis, :] + mu[np.newaxis, :] + centers[np.newaxis, :]
        return npFuture