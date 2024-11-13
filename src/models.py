import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats, lr=0.0001):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

class TranAD_without_sigmoid(nn.Module):
    def __init__(self, feats, lr=0.0001):
        super(TranAD_without_sigmoid, self).__init__()
        self.name = 'TranAD_without_sigmoid'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats))

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

class USAD_modified(nn.Module):
    def __init__(self, feats):
        super(USAD_modified, self).__init__()
        self.name = 'USAD_modified'
        self.lr = 0.0001
        self.n_feats = feats
        self.batch = 16
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = self.encoder(g)
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1, ae2, ae2ae1

# Proposed Model + Tcn_Local + Tcn_Global + Callback + Transformer + MAML
class DTAAD(nn.Module):
    def __init__(self, feats, lr=0.0001):
        super(DTAAD, self).__init__()
        self.name = 'DTAAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)  # K=3&4 (Batch, output_channel, seq_len)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers1 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers=1)  # only one layer
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c												# src is (batchsize, features, window_size)
        g_atts = self.g_tcn(src2)									# g_atts: (batchsize, features, window_size)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)	# src2: (window_size, batchsize, features)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, src):												# src: (batchsize, features, window_size)
        l_atts = self.l_tcn(src) 										# l_atts: (batchsize, features, window_size)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)		# src1: (window_size, batchsize, features)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        c1 = z1 + self.fcn(z1)											# c1: (window_size, batchsize, features)
        x1 = self.decoder1(c1.permute(1, 2, 0))							# input in decoder1: (batchsize, features, window_size)
        z2 = self.callback(src, x1)										# x1 has shape (batchsize, features, 1) src has shape (batchsize, features, window_size)
        c2 = z2 + self.fcn(z2)											# c2: (window_size, batchsize, features)
        x2 = self.decoder2(c2.permute(1, 2, 0))							# input in decoder2: (batchsize, features, window_size) | x2 (batchsize, features, 1)
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  				# return (batchsize, 1, features) for both x1 and x2

class DTAAD_without_sigmoid(nn.Module):
    def __init__(self, feats, lr=0.0001):
        super(DTAAD_without_sigmoid, self).__init__()
        self.name = 'DTAAD_without_sigmoid'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)  # K=3&4 (Batch, output_channel, seq_len)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers1 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers=1)  # only one layer
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Linear(self.n_window, 1)
        self.decoder2 = nn.Linear(self.n_window, 1)

    def callback(self, src, c):
        src2 = src + c												# src is (batchsize, features, window_size)
        g_atts = self.g_tcn(src2)									# g_atts: (batchsize, features, window_size)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)	# src2: (window_size, batchsize, features)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, src):												# src: (batchsize, features, window_size)
        l_atts = self.l_tcn(src) 										# l_atts: (batchsize, features, window_size)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)		# src1: (window_size, batchsize, features)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        c1 = z1 + self.fcn(z1)											# c1: (window_size, batchsize, features)
        x1 = self.decoder1(c1.permute(1, 2, 0))							# input in decoder1: (batchsize, features, window_size)
        z2 = self.callback(src, x1)										# x1 has shape (batchsize, features, 1) src has shape (batchsize, features, window_size)
        c2 = z2 + self.fcn(z2)											# c2: (window_size, batchsize, features)
        x2 = self.decoder2(c2.permute(1, 2, 0))							# input in decoder2: (batchsize, features, window_size) | x2 (batchsize, features, 1)
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  				# return (batchsize, 1, features) for both x1 and x2


