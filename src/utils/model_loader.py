import torch
from src.utils.s3_manager import download_from_s3
from src.model.logbert import LogBERT
from src.utils.config_loader import load_config
import yaml

def get_loaded_model():
    # Load config
    cfg = load_config()

    # Download from S3
    bucket = cfg['s3']['bucket']
    model_key = cfg['s3']['model_key']
    center_key = cfg['s3']['center_key']
    vocab_key = cfg['s3']['vocab_key']

    download_from_s3(bucket, model_key, "model.pth")
    download_from_s3(bucket, center_key, "center_vector.pt")
    download_from_s3(bucket, vocab_key, "log_key_to_id.pth")

    # Load model
    vocab = torch.load("log_key_to_id.pth", map_location="cpu")
    vocab_size = max(vocab.values()) + 1
    model = LogBERT(vocab_size=vocab_size).to("cpu")
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    center_vector = torch.load("center_vector.pt", map_location="cpu")

    return model, center_vector, vocab
