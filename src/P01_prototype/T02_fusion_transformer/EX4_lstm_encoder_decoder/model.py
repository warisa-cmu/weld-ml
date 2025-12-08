import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, source):
        # source: [batch_size, src_seq_len]
        embedded = self.embedding(source)  # [batch_size, src_seq_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        # input: [batch_size] (just the current token)
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden, cell


INPUT_DIM = 1000  # vocab size for source language
OUTPUT_DIM = 1000  # vocab size for target language
EMB_DIM = 256
HIDDEN_DIM = 512

encoder = EncoderLSTM(INPUT_DIM, EMB_DIM, HIDDEN_DIM)
decoder = DecoderLSTM(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM)

# Dummy input
src = torch.randint(0, INPUT_DIM, (32, 10))  # batch of 32, sentence length 10
trg = torch.randint(0, OUTPUT_DIM, (32,))  # next token(s)

hidden, cell = encoder(src)
output, hidden, cell = decoder(trg, hidden, cell)
