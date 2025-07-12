# src/api/main.py

from fastapi import FastAPI
from utils.s3_manager import S3Manager
from utils.config_loader import load_config
from src.inference_pipeline import infer_logs
from src.training_pipeline import train_model
from utils.config_loader import load_config


app = FastAPI()
config = load_config()
s3 = S3Manager(config['s3'])

@app.get("/")
async def root():
    return {"message": "LogBERT Anomaly Detection API"}

@app.get("/model/status")
async def model_status():
    return {"latest_model": config['s3']['latest_model_key'], "previous_model": config['s3']['previous_model_key']}

@app.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model)
    return {"status": "Training triggered in background"}

@app.post("/infer_batch")
def infer_batch(request: dict):
    logs = request.get("logs", [])
    cfg = load_config()
    anomalies = infer_logs(logs, entropy_threshold=cfg['thresholds']['entropy'], distance_threshold=cfg['thresholds']['distance'])
    return {"anomalies": anomalies}
