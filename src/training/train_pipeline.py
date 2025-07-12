import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import resample
from src.utils.s3_manager import upload_to_s3
from src.utils.config_loader import load_config
from core.prometheus_metrics import training_runs
from src.model.logbert import LogBERT, sequence_entropy
from src.data.data_process import parse_train_logs

class LogKeyDataset(Dataset):
    def __init__(self, sequences, vocab_size, mask_prob=MASK_PROB):
        self.sequences = sequences
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        seq = item["sequence"]
        input_seq, mask_positions, mask_labels = [], [], []
        mask_candidates = [i for i, t in enumerate(seq) if t != DIST_TOKEN_ID]
        num_to_mask = max(1, int(len(mask_candidates) * self.mask_prob))
        masked_indices = random.sample(mask_candidates, min(num_to_mask, len(mask_candidates)))

        for i, tok in enumerate(seq):
            if i in masked_indices:
                mask_positions.append(i)
                mask_labels.append(tok)
                rand = random.random()
                if rand < 0.8:
                    input_seq.append(self.mask_token_id)
                elif rand < 0.9:
                    input_seq.append(random.randint(1, self.vocab_size - 1))
                else:
                    input_seq.append(tok)
            else:
                input_seq.append(tok)

        label = 1 if any(u in anomalous_vm_ids for u in item['uuids']) else 0
        return torch.tensor(input_seq), torch.tensor(mask_positions), torch.tensor(mask_labels), label

def collate_fn(batch):
    input_seqs, mask_pos_list, mask_lbl_list, labels = zip(*batch)
    padded_inputs = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    return padded_inputs, mask_pos_list, mask_lbl_list, torch.tensor(labels)

def train_model():
    config = load_config()
    LOG_FILE = config["data"]["log_file"]
    LABEL_FILE = config["data"]["label_file"]
    ENTROPY_THRESHOLD = config["entropy_threshold"]
    EPOCHS = config["training"]["epochs"]
    # Preprocess logs
    train_sequences, eval_sequences, vocab_size = parse_train_logs(LOG_FILE, LABEL_FILE)

    # Dataset and loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = LogKeyDataset(train_sequences, vocab_size)
    eval_dataset = LogKeyDataset(eval_sequences, vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = LogBERT(vocab_size, embed_dim=EMBED_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Compute center vector
    model.eval()
    with torch.no_grad():
    all_embeddings = []
    for input_seqs, _, _, _ in train_loader:
        input_seqs = input_seqs.to(device)
        _, dist_embeds = model(input_seqs)
        all_embeddings.append(dist_embeds)
    center_vector = torch.cat(all_embeddings).mean(dim=0)

    from src.metrics_utils import (training_runs, training_epochs, training_loss,training_precision, training_recall, 
                                   training_f1_score, training_roc_auc, training_gradient_norm,training_learning_rate)
    # Training loop
    
    for epoch in range(EPOCHS):
        training_epochs.inc()
        model.train()
        total_loss = 0
        for input_seqs, mask_pos_list, mask_lbl_list, labels in train_loader:
            input_seqs = input_seqs.to(device)
            logits, dist_embeds = model(input_seqs)
            loss = 0
            for b in range(len(mask_pos_list)):
                if len(mask_pos_list[b]) == 0:
                    continue
                pos_tensor = mask_pos_list[b].to(device)
                true_tensor = mask_lbl_list[b].to(device)
                pred_logits = logits[b, pos_tensor]
                loss += criterion(pred_logits, true_tensor)
            vhm_loss = torch.mean(torch.norm(dist_embeds - center_vector, dim=1) ** 2)
            total = loss + ALPHA * vhm_loss
            optimizer.zero_grad()
            total.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            training_gradient_norm.set(total_norm)      
            optimizer.step()
            total_loss += total.item()
            scheduler.step()
            training_loss.set(total_loss)
            lr = scheduler.get_last_lr()[0]
            training_learning_rate.set(lr)
    # Save artifacts
    torch.save(model.state_dict(), "artifacts/trained_logbert.pth")
    torch.save(center_vector, "artifacts/center_vector.pt")
    torch.save(vocab_size, "artifacts/vocab_size.pth")
    upload_to_s3("artifacts/trained_logbert.pth", "models/latest/trained_logbert.pth")
    upload_to_s3("artifacts/center_vector.pt", "models/latest/center_vector.pt")
    upload_to_s3("artifacts/vocab_size.pth", "models/latest/vocab_size.pth")
    training_runs.inc()
    print("✅ Training complete and model uploaded to S3")

    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for input_seqs, mask_pos_list, mask_lbl_list, labels in eval_loader:
            input_seqs = input_seqs.to(device)
            logits, _ = model(input_seqs)
            for b in range(len(mask_pos_list)):
                if len(mask_pos_list[b]) == 0:
                    continue
                pos_tensor = mask_pos_list[b].to(device).long()
                entropy = sequence_entropy(logits[b, pos_tensor])
                all_scores.append(entropy)
                all_labels.append(labels[b].item())

    roc_auc = roc_auc_score(all_labels, all_scores)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recall, precision)
    preds = [1 if s >= 0.3 else 0 for s in all_scores]
    acc = accuracy_score(all_labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', zero_division=0)

    training_precision.set(p)
    training_recall.set(r)
    training_f1_score.set(f1)
    training_roc_auc.set(roc_auc)
