import torch

# ----- Generate Melody -----
class MelodyGenerator:

    def __init__(self, model, tokenizer, device, max_length=50):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = max_length
        

    def _get_sequence_tokenized(self, start_sequence, tokenizer):
        input_sequence = tokenizer.texts_to_sequences([start_sequence])
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).to(self.device)
        return input_tensor
    
    
    def _get_note_with_highest_score(self, output):
        latest_output = output[:, -1, :]  # output:  (batch_size, sequence_length, vocab_size)
        predicted_note_index = torch.argmax(latest_output, dim=1)
        predicted_note = predicted_note_index[0].unsqueeze(0).unsqueeze(-1)
        return predicted_note


    def _append_predicted_note(self, input_tensor, predicted_note):
        return torch.cat((input_tensor, predicted_note), dim=1) # dim=1，表示按照序列維度


    def _decode_generated_sequence(self, generated_sequence, tokenizer):
        generated_melody = tokenizer.sequences_to_texts(generated_sequence.cpu().numpy())[0].split()
        # [generated_sequence_array[0].tolist()] is a list of lists.
        return generated_melody
    

    def generate(self, start_sequence, tokenizer):
        input_tensor = self._get_sequence_tokenized(start_sequence, tokenizer)
        num_notes_to_generate = self.max_length - len(input_tensor[0])
        for _ in range(num_notes_to_generate):
            self.model.eval() # predict
            with torch.no_grad():    # 不計算梯度(因為沒有要訓練)
                output = self.model(input_tensor, input_tensor)
                predicted_note = self._get_note_with_highest_score(output)
                input_tensor = self._append_predicted_note(input_tensor, predicted_note)
                generated_melody = self._decode_generated_sequence(input_tensor, tokenizer)
        return generated_melody


# input_tensor = melody_generator._get_sequence_tokenized(start_sequence, preprocessor.tokenizer)


# output = melody_generator.model(input_tensor, input_tensor)
# predicted_note = melody_generator._get_note_with_highest_score(output)
# input_tensor = melody_generator._append_predicted_note(input_tensor, predicted_note)
# melody_generator._decode_generated_sequence(input_tensor, preprocessor.tokenizer)
