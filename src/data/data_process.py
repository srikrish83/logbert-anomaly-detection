import re
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from src.utils.config_loader import load_config

config = load_config()
PREPROC_CFG = config["preprocessing"]
DRAIN_CFG = config["drain3"]

SEQ_WINDOW = PREPROC_CFG["seq_window"]
DIST_TOKEN_ID = PREPROC_CFG["dist_token_id"]
MAX_SEQ_LEN = PREPROC_CFG["max_seq_len"]

def setup_drain3():
    """Setup Drain3 miner directly from YAML config."""
    cfg = TemplateMinerConfig()
    for key, value in DRAIN_CFG.items():
        cfg.set("DEFAULT", key, str(value))
    return TemplateMiner(config=cfg)

def extract_uuids(line):
    return {m.lower() for m in re.findall(r'[0-9a-fA-F-]{36}', line)}

def mask_uuid_ip_path(line):
    line = re.sub(r'[0-9a-fA-F-]{36}', '<UUID>', line)
    line = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', line)
    line = re.sub(r'/[^\s"]+', '<PATH>', line)
    return line

def preprocess_train_logs(log_file, label_file):
    # === Parse Logs ===
    template_miner = setup_drain3()
    log_key_to_id = {}  
    next_log_key_id = 1
    sequences = []
    session_to_id = {}
    session_logs = {}
    next_session_id = 1

    with open(log_file, 'r') as f:
        for line in f:
            uuids_found = extract_uuids(line)
            masked_line = mask_uuid_ip_path(line)
            result = template_miner.add_log_message(masked_line.strip())
            if not result:
                continue
            template = result["template_mined"]
            if template not in log_key_to_id:
                log_key_to_id[template] = next_log_key_id
                next_log_key_id += 1
            log_key_id = log_key_to_id[template]

            for uuid in uuids_found:
                if uuid not in session_to_id:
                    session_to_id[uuid] = next_session_id
                    next_session_id += 1
                sid = session_to_id[uuid]
                session_logs.setdefault(sid, []).append((log_key_id, uuids_found))

    for sid, log_list in session_logs.items():
        tokens = [DIST_TOKEN_ID] + [entry[0] for entry in log_list]
        all_uuids = set(u for _, ulist in log_list for u in ulist)
        for i in range(0, len(tokens) - SEQ_WINDOW):
            seq_slice = tokens[i:i + SEQ_WINDOW + 1]
            if len(seq_slice) > MAX_SEQ_LEN:
                seq_slice = seq_slice[:MAX_SEQ_LEN]
            sequences.append({"sequence": seq_slice, "uuids": list(all_uuids), "session_id": sid})
    return sequences

def parse_train_logs(log_file, label_file):
    sequences = preprocess_train_logs(log_file, label_file)

    with open(label_file, 'r') as f:
        anomalous_vm_ids = set(line.strip().lower() for line in f if re.match(r'^[0-9a-f-]{36}$', line.strip()))

    anomalous_sequences = [s for s in sequences if any(u in anomalous_vm_ids for u in s['uuids'])]
    normal_sequences = [s for s in sequences if not any(u in anomalous_vm_ids for u in s['uuids'])]
    rng = np.random.default_rng(seed=42)

    # Randomly select exactly 12327 indices
    num_eval_normals = len(anomalous_sequences)
    normal_indices = np.arange(len(normal_sequences))
    eval_indices = rng.choice(normal_indices, size=num_eval_normals, replace=False)

    # Split into eval_normals and remaining train_normals
    eval_normals = [normal_sequences[i] for i in eval_indices]
    train_sequences = [s for i, s in enumerate(normal_sequences) if i not in eval_indices]
    eval_sequences = anomalous_sequences + eval_normals

    vocab_size = max(max(seq['sequence']) for seq in sequences) + 1

    return train_sequences, eval_sequences, vocab_size

def preprocess_infer_logs(logs, log_key_to_id):
    """Preprocess logs for inference."""
    template_miner = setup_drain3()
    sequences = []
    current_tokens = [DIST_TOKEN_ID]
    raw_lines = []
    anomalous_outputs = []
    session_to_id = {}
    session_logs = {}
    next_session_id = 1

    for line in logs:
        masked = mask_uuid_ip_path(line.strip())
        uuids = extract_uuids(line)
        result = template_miner.add_log_message(masked)
        if not result:
            continue
        template = result["template_mined"]
        if template not in log_key_to_id:
            continue
        key_id = log_key_to_id[template]
        for uuid in uuids:
            if uuid not in session_to_id:
                session_to_id[uuid] = next_session_id
                next_session_id += 1
            sid = session_to_id[uuid]
            session_logs.setdefault(sid, []).append((key_id, line))

    # === Generate sequences per UUID session ===
    sequences = []
    seen_sequences = set()
    for sid, entries in session_logs.items():
        tokens = [DIST_TOKEN_ID] + [x[0] for x in entries]
        raw_lines = [x[1] for x in entries]
        for i in range(0, len(tokens) - SEQ_WINDOW):
            seq_slice = tokens[i:i + SEQ_WINDOW + 1]
            line_slice = raw_lines[i:i + SEQ_WINDOW + 1]
            if len(seq_slice) > MAX_SEQ_LEN:
                seq_slice = seq_slice[:MAX_SEQ_LEN]
                line_slice = line_slice[:MAX_SEQ_LEN]
            seq_key = tuple(seq_slice)
            if seq_key in seen_sequences:
              continue  # skip duplicate
            seen_sequences.add(seq_key)
            sequences.append((seq_slice, line_slice))

    return sequences