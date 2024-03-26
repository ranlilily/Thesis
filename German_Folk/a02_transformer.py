import torch
import torch.nn as nn

# ----- Position Encoding -----
def position_encoding(num_pos, d_model):
    position = torch.arange(num_pos).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-torch.log(torch.tensor(10000.0)) / d_model))
    angles = position * div_term
    pos_encoding = torch.zeros(num_pos, d_model)
    pos_encoding[:, 0::2] = torch.sin(angles)
    pos_encoding[:, 1::2] = torch.cos(angles)
    return pos_encoding.unsqueeze(0)  # Add batch dimension

# ----- 確認 padding 遮罩 （布林） -----
def key_padding_mask(seq, pad_token=0):
    return (seq == pad_token)

'''
原理:這個函式通過比較序列中的元素與填充標記,生成一個布林遮罩。如果元素等於填充標記,則對應的遮罩值為True,否則為False。
返回值：函式返回一個布林遮罩,其形狀與輸入序列相同,其中填充位置為True,非填充位置為False。
'''

# ----- Look ahead mask (遮住後面尚未出現的資訊) -----
def look_ahead_mask(dim):
    return nn.Transformer.generate_square_subsequent_mask(dim)

'''
這個函式生成了一個用於自注意力機制的遮罩，該遮罩確保了在每個時間步上模型只能關注到當前及之前的元素，
而不能關注到未來的元素。這對於自注意力機制的正確運作至關重要。

- 參數 dim:表示生成遮罩的維度。這個參數通常是序列的長度。
- 原理:這個函式調用了PyTorch中nn.Transformer.generate_square_subsequent_mask方法來生成遮罩。這個方法生成了一個方形的上三角遮罩矩陣，
       對角線以下的元素為True,表示模型在計算每個位置的注意力時只能關注到當前及之前的元素。
- 返回值：函式返回一個遮罩，形狀為 (dim, dim)，其中 dim 是遮罩的維度。
'''

# ----- Build Transformer Model -----
class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward, vocab_size_padding,
                 num_encoder_layers, num_decoder_layers, device):
        super(TransformerModel, self).__init__()  
        self.d_model = d_model
        self.device = device
        self.embedding = nn.Embedding(vocab_size_padding, d_model).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 
                                                        dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, 
                                                        dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(d_model, vocab_size_padding)



    def forward(self, src, tgt):
        src_padding_mask = key_padding_mask(src).to(self.device)
        tgt_padding_mask = key_padding_mask(tgt).to(self.device)
        tgt_mask = look_ahead_mask(tgt.size(-1)).to(self.device)  
        scale_factor = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=self.device))

        x = self.embedding(src) 
        x *= scale_factor
        x += position_encoding(src.size(-1), self.d_model).to(self.device)
        x = self.dropout(x)
        enc_output = self.encoder(x, src_key_padding_mask=src_padding_mask)

        y = self.embedding(tgt)
        y *= scale_factor
        y += position_encoding(tgt.size(-1), self.d_model).to(self.device)
        y = self.dropout(y)
        dec_output = self.decoder(y, enc_output, tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        output = self.final_layer(dec_output)
        return output

# scale_factor 是一個縮放因子，用於縮放嵌入向量的值。它根據 d_model 的平方根來縮放，以防止在多頭注意力機制中梯度消失或爆炸。
# self.final_layer 是一個全連接層，用於將解碼器的輸出映射到目標空間的維度上。
