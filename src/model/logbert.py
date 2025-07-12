import torch
import torch.nn as nn

def sinusoidal_positional_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class LogBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.pos_encoding = sinusoidal_positional_encoding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        pos_embed = self.pos_encoding[:, :input_ids.size(1), :].to(input_ids.device)
        x = self.embedding(input_ids) + pos_embed
        pad_mask = input_ids == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        logits = self.classifier(x)
        return logits, x[:, 0, :]

def sequence_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()