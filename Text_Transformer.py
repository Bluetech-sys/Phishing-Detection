import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
import macros
import transformers 
import torch.nn as nn
from transformers import AutoModel, AutoConfig,PreTrainedTokenizerBase
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
import pandas as pd

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(macros.SEED)


def chunk_tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int = 512,
    stride: int = 128,
    truncation: bool = False,
):
    """
    Tokenize a single text into one or more chunks using a sliding window.
    Returns list of tokenized encodings (dicts).
    """
    max_len = 512
    stride = 256

    text_tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(text_tokens)

    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + max_len
        chunk = token_ids[start:end]
        chunks.append(chunk)
        if end >= len(token_ids):
            break
        start += stride  # move start by stride for overlap
    return chunks


def aggregate_chunk_predictions(orig_ids: List[Any], logits: np.ndarray, method: str = "majority"):
    """
    orig_ids: list of original example ids (one per chunk)
    logits: array shape (num_chunks, num_labels)
    method: 'majority' or 'mean'
    Returns:
        agg_preds: list of (orig_id, pred_label, pred_prob)
    """
    assert method in ("majority", "mean")
    grouped = defaultdict(list)
    for oid, logit in zip(orig_ids, logits):
        grouped[oid].append(logit)

    results = []
    for oid, logs in grouped.items():
        arr = np.stack(logs, axis=0)  # (num_chunks_for_this_example, num_labels)
        probs = torch.softmax(torch.from_numpy(arr), dim=-1).numpy()  # convert to probs
        if method == "mean":
            mean_prob = probs.mean(axis=0)
            pred_label = int(mean_prob.argmax())
            pred_prob = float(mean_prob.max())
        else:  # majority
            # for each chunk take argmax, then majority vote
            chunk_preds = [int(p.argmax()) for p in probs]
            most_common, count = Counter(chunk_preds).most_common(1)[0]
            pred_label = int(most_common)
            # approximate confidence: fraction of chunks voting for winner
            pred_prob = float(count / len(chunk_preds))
        results.append((oid, pred_label, pred_prob))
    return results

# -----------------------------
# Data loading helper
# -----------------------------
def load_csvs(train_csv: str, valid_csv: str):
    """
    Loads CSV files into a HuggingFace DatasetDict
    Expects columns: text,label
    """
    data_files = {}
    if os.path.exists(train_csv):
        data_files["train"] = train_csv
    if os.path.exists(valid_csv):
        data_files["validation"] = valid_csv
    if not data_files:
        raise FileNotFoundError("No CSVs found at the provided paths.")
    ds = load_dataset("csv", data_files=data_files)
    # Optionally check columns
    if "label" not in ds[list(ds.keys())[0]].column_names:
        raise ValueError("CSV must contain 'label' column with 0/1 values.")
    return ds

# -----------------------------
# Metrics
# -----------------------------
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    prec = precision_metric.compute(predictions=preds, references=labels, zero_division=0)["precision"]
    rec = recall_metric.compute(predictions=preds, references=labels, zero_division=0)["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, zero_division=0)["f1"]
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
  

class CustomDebertaClassifier(nn.Module):
    def __init__(self, model_name: str,
                num_labels: int, 
                hidden_layer_sizes=macros.Classificaton_Hidden_Layers, 
                dropout_prob=macros.DROP_OUT_PROB):
        super().__init__()
        # Load pretrained DeBERTa model backbone
        self.deberta = DebertaModel.from_pretrained(model_name)
        
        # Hidden size from DeBERTa config
        hidden_size = self.deberta.config.hidden_size
        
        # Build classification head dynamically
        layers = []
        input_size = hidden_size
        total_layers = len(hidden_layer_sizes)
        for hidden_size_ in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size_))
            layers.append(nn.ReLU())
            # if hidden_size>= total_layers -3:
            #     layers.append(nn.Sigmoid())
            # else:
            #     layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size_
        
        # Final classification layer
        layers.append(nn.Linear(input_size, num_labels))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Forward pass through DeBERTa
        outputs = self.deberta(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        # Take the [CLS] token (first token) output
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

    def save_pretrained(self, save_path: str):
        """Save model weights + config"""
        torch.save(self.state_dict(), f"{save_path}/pytorch_model.bin")
        self.deberta.config.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_path: str, 
                        num_labels: int, 
                        hidden_layer_sizes=macros.Classificaton_Hidden_Layers, 
                        dropout_prob=macros.DROP_OUT_PROB):
        """Load architecture and weights"""
        config = AutoConfig.from_pretrained(model_path)
        model = cls(
            model_name=model_path,
            num_labels=num_labels,
            hidden_layer_sizes=hidden_layer_sizes,
            dropout_prob=dropout_prob
        )
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

# -----------------------------
# Dataset wrapper
# -----------------------------
class PhishDataset(Dataset):
    """
    Dataset that loads data from:
    - a HF Dataset (with 'text' and 'label')
    - OR a path to a CSV file
    - OR a pandas DataFrame
    and tokenizes with chunking.
    For training each chunk is a separate example (retaining same label).
    """

    def __init__(
        self,
        data_source,  # can be HF Dataset, CSV filepath (str), pandas DataFrame, or list of dicts
        tokenizer: PreTrainedTokenizerBase,
        mode: str = "train",
        max_length=macros.MAX_LENGTH,
        stride=macros.STRIDE,
    ):
        assert mode in ("train", "eval", "predict")
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.mode = mode

        # Load or convert data_source to HF Dataset
        if isinstance(data_source, str):
            # assume CSV file path
            hf_dataset = load_dataset("csv", data_files=data_source)["train"]
        elif isinstance(data_source, pd.DataFrame):
            hf_dataset = HFDataset.from_pandas(data_source.reset_index(drop=True))
        elif isinstance(data_source, list):
            # list of dicts
            hf_dataset = HFDataset.from_list(data_source)
        elif isinstance(data_source, HFDataset):
            hf_dataset = data_source
        else:
            raise ValueError("data_source must be a path to CSV, pandas DataFrame, list of dicts, or Hugging Face Dataset")

        # Process each example
        for item in hf_dataset:
            text = item["text"]
            label = item.get("label", None)
            chunks = chunk_tokenize_text(tokenizer, text, max_length=max_length, stride=stride)
            # For each chunk, store text->chunk mapping
            for chunk in chunks:
                entry = {
                    "input_ids": chunk["input_ids"],
                    "attention_mask": chunk["attention_mask"],
                }
                if label is not None:
                    entry["label"] = int(label)
                if "id" in item:
                    entry["orig_id"] = item["id"]
                self.examples.append(entry)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        # Convert lists to torch tensors is done by collator
        return item

  