# src/model/utils.py

import torch
import torch.nn.functional as F

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()

def compute_distance(embedding, center_vector):
    return torch.norm(embedding - center_vector, dim=1).mean().item()
