from prometheus_client import Counter, Gauge, Histogram

# Training metrics
training_runs = Counter("training_runs_total", "Total number of training runs")
training_epochs = Counter("training_epochs_total", "Total number of epochs completed")
training_loss = Gauge("training_loss", "Training loss per epoch")
training_precision = Gauge("training_precision", "Precision after evaluation")
training_recall = Gauge("training_recall", "Recall after evaluation")
training_f1_score = Gauge("training_f1_score", "F1 Score after evaluation")
training_roc_auc = Gauge("training_roc_auc", "ROC-AUC after evaluation")
training_gradient_norm = Gauge("training_gradient_norm", "Gradient norm during training")
training_learning_rate = Gauge("training_learning_rate", "Learning rate during training")

# Inference metrics
inference_runs = Counter("inference_runs_total", "Total number of inference runs")
anomalies_detected = Counter("inference_anomalies_detected", "Number of anomalies detected during inference")

inference_entropy_mean = Gauge("inference_entropy_mean", "Mean entropy of all sequences")
inference_entropy_max = Gauge("inference_entropy_max", "Max entropy of all sequences")
inference_distance_mean = Gauge("inference_distance_mean", "Mean distance of all sequences")
inference_distance_max = Gauge("inference_distance_max", "Max distance of all sequences")

# === Histograms ===
inference_latency_seconds = Histogram("inference_latency_seconds", "Total inference latency in seconds")


def start_metrics_server(port=8000):
    """Start Prometheus metrics server for scraping"""
    start_http_server(port)
