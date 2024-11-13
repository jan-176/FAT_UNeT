import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model, hidden, drop_prob=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, hidden)
    self.linear2 = nn.Linear(hidden, d_model)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(p=drop_prob)

  def forward(self, x):
    x = self.linear1(x)
    x = self.sigmoid(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x
 
class LearnableSPE(nn.Module):
  def __init__(self, d_model: int, d_hid: int, dropout: float = 0.1, max_len: int = 512):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.ffn = PositionwiseFeedForward(d_model, d_hid)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
    """
    x = x.transpose(0, 1) # after transpose x has shape [seq_len, batch_size, embedding_dim]
    self.pe_out = self.ffn(self.pe) # Add nn so that encoding becomes trainable
    x = x + self.pe_out[:x.size(0)]
    x = x.transpose(0, 1) # after transpose x has shape [batch_size, seq_len, embedding_dim]
    return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, n_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(model_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_hidden_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src2 = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, n_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(model_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_hidden_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(tgt2, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    

class FAT(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim, dropout=0.1):
        super(FAT, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc = 10
        self.n_dec = 10
        self.n_window = self.n_enc + self.n_dec
        self.batch = 128
        self.name="FAT"

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = LearnableSPE(d_model, d_model, dropout)

        self.decoder = nn.Linear(input_dim, d_model)
        self.pos_decoder = LearnableSPE(d_model, d_model, dropout)

        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, ff_hidden_dim, dropout) for _ in range(num_encoder_layers)])
        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads, ff_hidden_dim, dropout) for _ in range(num_decoder_layers)])

        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt):
        # Embedding and Positional Encoding
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.decoder(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)

        # Encoder
        for layer in self.transformer_encoder:
            src = layer(src)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        # Decoder
        for layer in self.transformer_decoder:
            tgt = layer(tgt, src, tgt_mask=tgt_mask)

        output = self.fc(tgt)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.double().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



