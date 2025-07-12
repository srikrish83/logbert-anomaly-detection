import time
import json
from kafka import KafkaProducer
from src.utils.config_loader import load_config

# === CONFIG ===
KAFKA_BROKER = "localhost:9092"  # Update if needed
TOPIC_NAME = "logs-input"        # Topic for inference pipeline
LOG_FILE_PATH = "sample_logs.log"  # Path to your test log file
BATCH_SIZE = 10                  # Number of lines per message
INTERVAL_SEC = 5                 # Wait time between batches (seconds)
ENCODING = "utf-8"

def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode(ENCODING),
        retries=5
    )

def produce_logs(file_path, batch_size, interval_sec):
    producer = create_producer()
    print(f"🚀 Kafka producer started. Pushing logs from {file_path} to topic '{TOPIC_NAME}'")

    with open(file_path, "r", encoding=ENCODING) as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if len(batch) >= batch_size:
                # Send batch
                producer.send(TOPIC_NAME, {"logs": batch})
                producer.flush()
                print(f"✅ Sent batch of {len(batch)} logs to Kafka")
                batch = []
                time.sleep(interval_sec)

        # Send any remaining lines
        if batch:
            producer.send(TOPIC_NAME, {"logs": batch})
            producer.flush()
            print(f"✅ Sent final batch of {len(batch)} logs to Kafka")

    producer.close()
    print("🎯 Finished producing logs.")

if __name__ == "__main__":
    produce_logs(LOG_FILE_PATH, BATCH_SIZE, INTERVAL_SEC)
