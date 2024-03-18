import torch
from torch.utils.data import TensorDataset
import tensorflow as tf
# from keras.preprocessing.text import Tokenizer (老師原始的，跑不動，Pytorch 沒有類似的)
from tensorflow.keras.preprocessing.text import Tokenizer


# ----- Tokenize and Encoding -----
class MelodyPreprocessor:
    
    ## This class takes symbolized melodies, tokenizes and encodes them, and prepares
    ## datasets for training sequence-to-sequence models.

    def __init__(self, dataset):     
        self.dataset = dataset
        self.tokenizer = Tokenizer(filters="", lower=False)  # Tokenizer 是 Keras 中用於文本標記化的類別，它可以將文本轉換成序列數據 (類似 dictionary)
        self.seq_length = None
        self.vocab_size = None
        self.data_size = None


    def create_training_dataset(self):    
        tokenized_melodies = self._tokenize_and_encode_melodies(self.dataset) # 將原始的旋律數據集進行標記化和編碼
        self._set_max_melody_length(tokenized_melodies) # 記錄數據集中最長的旋律的長度
        self._set_number_of_unique_tokens()   # vocab_size
        seq_pairs = self._create_pad_sequence_pairs(tokenized_melodies) # 每個序列填充到與最長旋律相同的長度
        return seq_pairs


    def _tokenize_and_encode_melodies(self, melodies):
        self.tokenizer.fit_on_texts(melodies)  # 建立一個詞彙表，並為每個單詞（在這個情況下是音符）分配一個唯一的數字標籤。 (類似 dictionary)
        tokenized_melodies = self.tokenizer.texts_to_sequences(melodies)
        return tokenized_melodies

    def _set_max_melody_length(self, melodies):
        self.seq_length = max([len(melody) for melody in melodies])

    # 找到的最大旋律長度，這樣就可以確保所有的輸入序列都可以被填充到相同的長度，以便進行批處理。


    def _set_number_of_unique_tokens(self):   # 計算 Voc 的大小
        self.vocab_size = len(self.tokenizer.word_index)


    def _pad_sequence(self, sequence):
        # make each sequence the same length
        # empty spot coded as zero
        return sequence + [0] * (self.seq_length - len(sequence))
    
    # 這個方法 _pad_sequence 用於將序列（sequence）填充到相同的長度，以確保所有的序列都具有相同的長度。
    # 這在訓練序列模型時非常重要，因為模型需要批次處理相同大小的輸入。


    def _create_pad_sequence_pairs(self, melodies):
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1 : i + 1]  # Shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        sequence_pairs = TensorDataset(torch.tensor(input_sequences, dtype=torch.long), 
                                       torch.tensor(target_sequences, dtype=torch.long))
        self.data_size = len(sequence_pairs)
        return sequence_pairs

    # 這個方法將每個旋律拆分成多個輸入-目標序列對，並對每個序列進行填充，從而創建了一個適合用於訓練 Transformer 模型的數據集                                       
