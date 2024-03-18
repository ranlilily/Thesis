import torch
import torch.nn as nn
import torch.optim as optim

import a02_transformer  
from a02_transformer import TransformerModel
from importlib import reload
reload(a02_transformer)
from a02_transformer import TransformerModel

from a01_melody_preprocessor import MelodyPreprocessor
from a02_transformer import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# ----- Train & Loss  -----
def train_each_step(training_batches, model, criterion, optimizer, device):
    model.train()  # Ensure the model is in training mode
    total_loss = 0
    for src_batch, tgt_batch in training_batches:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        output = model(src_batch, tgt_batch[:, :-1]) # Exclude the last element for prediction targets  ＃tgt_batch[:, :-1] 表示將目標序列中的最後一個標記（即 <eos> 標記）排除在外
        output = output.reshape(-1, output.shape[-1])  # Reshape for CrossEntropyLoss  ＃重新塑形為二維張量
        loss = criterion(output, tgt_batch[:, 1:].reshape(-1))  # Shift targets for loss calculation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(training_batches)
    return average_loss

# tgt_batch[:, 1:].reshape(-1) 表示將目標序列中的第一個標記（即 <sos> 標記）排除在外，用於計算損失。這樣做是為了對齊模型的輸出和目標序列。


# ----- 可忽略 -----
# if __name__ == "__main__":
#     preprocessor = MelodyPreprocessor("dataset.json", batch_size=32)
#     training_dataset_batches = preprocessor.create_training_dataset()
#     vocab_size_padding = preprocessor.vocab_size + 1

#     model = TransformerModel(d_model=64, nhead=2, dim_ff=64, dropout=0.1, 
#                              vocab_size_padding=vocab_size_padding, 
#                              num_encoder_layers=2, num_decoder_layers=2, device=device)
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.0002)

#     # epochs = 1000
#     epochs = 5
#     for epoch in range(epochs):
#         average_loss = train_each_step(training_dataset_batches, model, 
#                                        criterion, optimizer, device)
#         print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')


