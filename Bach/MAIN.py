import os
import music21 as m21
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import importlib as imp
from torch.utils.data import DataLoader


import a02_transformer
imp.reload(a02_transformer)


import a00_funs_make_symbol_seqs as fmseq
from a01_melody_preprocessor import MelodyPreprocessor
from a02_transformer import TransformerModel
from a04_melody_generator import MelodyGenerator
import a03_train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


## Parameters for Data Preprocessing
time_signature = '4/4'
beats_per_measure=4
step_duration = 0.25  # 0.25 = a 1/16 note 
acceptable_durations = np.arange(0.25, 8.1, 0.25) 


## Import Data and Prepare batches
songs = m21.corpus.search('bach', fileExtensions='xml')
melodies = fmseq.make_melody_symbol_sequences(songs, time_signature, 
                                              acceptable_durations)
preprocessor = MelodyPreprocessor(melodies)
training_dataset = preprocessor.create_training_dataset()
training_batches = DataLoader(training_dataset, shuffle=True,
                              batch_size=128)

print(preprocessor.vocab_size)
print(preprocessor.data_size)
print(preprocessor.seq_length)




## Model Specification and Training
vocab_size_padding = preprocessor.vocab_size + 1
model = TransformerModel(d_model=128, nhead=2, dim_feedforward=128, dropout=0.1, 
                         vocab_size_padding=vocab_size_padding, 
                         num_encoder_layers=6, num_decoder_layers=6, device=device)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# epochs = 200
# save_interval = 20
epochs = 5
save_interval =1
save_dir = 'generated'
start_sequence = ['C4-1.0', 'G4-1.0', 'E4-1.0', 'C4-1.0']

for epoch in range(epochs):
    average_loss = a03_train.train_each_step(training_batches, model, 
                                             criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')

    if epoch > 0 and (epoch + 1) % save_interval == 0:
        melody_generator = MelodyGenerator(model, preprocessor.tokenizer, device)
        new_melody = melody_generator.generate(start_sequence, preprocessor.tokenizer)
        np.savetxt(f"{save_dir}/{epoch + 1}.txt", new_melody, fmt='%s')



## Generation
start_sequence = ["C4-2.0", "G4-2.0", "E4-2.0", "D4-1.0", "C4-1.0"]
start_sequence = ["C4-2.0", "F4-2.0", "A4-1.0", "D5-0.5", "C5-0.5"]
melody_generator = MelodyGenerator(model, preprocessor.tokenizer, device)
new_melody = melody_generator.generate(start_sequence, preprocessor.tokenizer)
print(f"Generated melody: {new_melody}") 




## Save model
torch.save(model.state_dict(), 'model_state_dict.pth')
torch.save(optimizer.state_dict(), 'optimizer_state_dict.pth')


## Load model
model = TransformerModel(d_model=128, nhead=2, dim_feedforward=128, dropout=0.1, 
                         vocab_size_padding=vocab_size_padding, 
                         num_encoder_layers=2, num_decoder_layers=2, device=device)
model.load_state_dict(torch.load('model_state_dict.pth'))
optimizer.load_state_dict(torch.load('optimizer_state_dict.pth'))








