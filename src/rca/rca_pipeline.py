import json
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from kafka import KafkaConsumer
from queue import Queue
from src.utils.config_loader import load_config

config = load_config()
# Shared queue for Streamlit
rca_queue = Queue()

# Kafka setup
bootstrap_servers = config["kafka"]["bootstrap_servers"]
anomalies_topic = config["kafka"]["anomalies_topic"]

consumer = KafkaConsumer(
    anomalies_topic,
    bootstrap_servers=bootstrap_servers,
    group_id="rca-consumer-group",
    auto_offset_reset="latest",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

# Hugging Face API setup
tokenizer = AutoTokenizer.from_pretrained(config["huggingface"]["model_name"], use_auth_token=config["huggingface"]["api_token"])
model = AutoModelForCausalLM.from_pretrained(config["huggingface"]["model_name"], use_auth_token=config["huggingface"]["api_token"])
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def generate_rca(anomaly_text):
    prompt = (
        "You are an expert SRE. Analyze the following anomaly logs and provide a Root Cause Analysis (RCA):\n\n"
        f"{anomaly_text}\n\nRCA:"
    )
    result = llm_pipeline(prompt, max_length=1024, do_sample=False)[0]['generated_text']
    return result

def consume_and_process():
    print("📡 RCA pipeline listening to anomalies...")
    for message in consumer:
        anomaly = message.value.get("anomaly", "")
        print(f"⚡ New anomaly:\n{anomaly[:200]}...")
        rca_text = generate_rca(anomaly)
        rca_queue.put({"anomaly": anomaly, "rca": rca_text})

# Run in background thread
def start_rca_pipeline():
    t = threading.Thread(target=consume_and_process, daemon=True)
    t.start()
