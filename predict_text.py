import torch
import macros
import numpy as np
from transformers import AutoTokenizer
from Text_Transformer import chunk_tokenize_text, aggregate_chunk_predictions ,CustomDebertaClassifier

class Predict_phishing:
    def __init__(self,
                 model_path="./saved_model",
                 max_length=macros.MAX_LENGTH,
                 stride=macros.STRIDE,
                 aggregation=macros.AGGREGATION):

        self.model_path = model_path
        self.max_length = max_length
        self.stride = stride
        self.aggregation = aggregation

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = CustomDebertaClassifier.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        """Make prediction for text input."""
        chunks = chunk_tokenize_text(self.tokenizer, text,
                                     max_length=self.max_length,
                                     stride=self.stride)
        inputs = self.tokenizer.pad(chunks, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits.detach().cpu().numpy()

        orig_ids = [0] * logits.shape[0]
        agg = aggregate_chunk_predictions(orig_ids, logits, method=self.aggregation)
        _, pred_label, pred_prob = agg[0]

        label_names = {0: "not_phishing", 1: "phishing"}
        return {
            "label_id": pred_label,
            "label": label_names[pred_label],
            "confidence": pred_prob
        }
