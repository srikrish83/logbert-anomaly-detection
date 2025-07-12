import torch
import torch.nn.functional as F
from src.utils.model_loader import get_loaded_model
from src.data.data_process import preprocess_infer_logs
from src.model.logbert import sequence_entropy
from src.utils.monitoring import (
    inference_runs,
    inference_latency_seconds,
    anomalies_detected,
    inference_entropy_mean,
    inference_entropy_max,
    inference_distance_mean,
    inference_distance_max
)
import numpy as np

model, center_vector, log_key_to_id = get_loaded_model()

@inference_latency_seconds.time()
def infer_logs(logs, entropy_threshold, distance_threshold):
    sequences = preprocess_logs(logs, log_key_to_id)
    anomalies = []

    entropy_values = []
    distance_values = []

    with torch.no_grad():
        for seq_tokens, raw_lines in sequences:
            input_tensor = torch.tensor([seq_tokens], dtype=torch.long)
            logits, dist_embeds = model(input_tensor)
            entropy = sequence_entropy(logits[0])
            distance = torch.norm(dist_embeds - center_vector).item()
            entropy_values.append(entropy)
            distance_values.append(distance)

            if entropy >= entropy_threshold and distance > distance_threshold:
                anomalies.append("\n".join(raw_lines))
    # Update Prometheus metrics
    if entropy_values:
        inference_entropy_mean.set(np.mean(entropy_values))
        inference_entropy_max.set(np.max(entropy_values))
    if distance_values:
        inference_distance_mean.set(np.mean(distance_values))
        inference_distance_max.set(np.max(distance_values))
    
    anomalies_detected.inc(len(anomalies))
    return anomalies
