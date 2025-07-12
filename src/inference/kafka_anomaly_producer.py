from kafka import KafkaProducer
from src.utils.config_loader import load_config
import yaml
import json

cfg = load_config()
producer = KafkaProducer(bootstrap_servers=cfg['kafka']['bootstrap_servers'])

def send_anomalies_to_kafka(anomalies):
    output_topic = cfg['kafka']['output_topic']
    for anomaly in anomalies:
        producer.send(output_topic, json.dumps({"anomaly": anomaly}).encode())
    producer.flush()
