from kafka import KafkaConsumer
from utils.config_loader import load_config
from src.inference.kafka_anomaly_producer import send_anomalies_to_kafka
import requests
import json
import yaml
import time

def consume_and_infer():
    cfg = load_config()
    interval = cfg['kafka']['batch_interval']
    fastapi_url = cfg['fastapi']['url']
    topic = cfg['kafka']['input_topic']
    servers = cfg['kafka']['bootstrap_servers']

    consumer = KafkaConsumer(topic, bootstrap_servers=servers, auto_offset_reset='latest')
    batch = []
    last_flush = time.time()

    for msg in consumer:
        batch.append(msg.value.decode())
        if time.time() - last_flush >= interval:
            response = requests.post(f"{fastapi_url}/infer_batch", json={"logs": batch})
            anomalies = response.json().get("anomalies", [])
            if anomalies:
                send_anomalies_to_kafka(anomalies)
            batch.clear()
            last_flush = time.time()
