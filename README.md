# 🚀 LogBERT Anomaly Detection Framework

A fully automated end-to-end platform for **log anomaly detection and root cause analysis** using LogBERT. It supports:

✅ **Training** via API  
✅ **Real-time inference** over Kafka streams  
✅ **RCA generation** via Hugging Face LLMs  
✅ **Streamlit dashboard** with live popups  
✅ **Prometheus + Grafana monitoring**  
✅ **S3 model artifact management**  

---

## 📐 System Architecture

```ascii
+-----------------+        +-----------------+         +-----------------+
|                 |        |                 |         |                 |
|  Log Producers  +------->+   Kafka Broker   +-------->+  Inference API  |
| (Apps, Systems) |        | (anomalies_topic)|         | (FastAPI)       |
+-----------------+        +-----------------+         +--------+--------+
                                                                 |
                                                                 v
                                                       +-----------------+
                                                       |   Hugging Face  |
                                                       |   RCA LLM       |
                                                       +--------+--------+
                                                                |
                                                                v
                                                       +-----------------+
                                                       | Streamlit UI    |
                                                       | (Live RCA Popups|
                                                       +-----------------+

Prometheus scrapes metrics from:
- Training API
- Inference API
- Kafka Consumer lag
