import os
import torch
import macros
import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from Text_Transformer import (
    CustomDebertaClassifier,
    load_csvs,
    PhishDataset,
    compute_metrics,
    chunk_tokenize_text
)

class Trainer:
    def __init__(self,
                 model_name=macros.MODEL_NAME,
                 num_labels=macros.NUM_LABELS,
                 max_length=macros.MAX_LENGTH,
                 stride=macros.STRIDE,
                 epochs=macros.EPOCHS,
                 batch_size=macros.BATCH_SIZE,
                 lr=macros.LR,
                 output_dir=macros.OUTPUT_DIR):

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.stride = stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.output_dir = output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # ✅ Conditional model loading
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            print(f"🔹 Found existing fine-tuned model at {model_path}. Loading...")
            self.model = CustomDebertaClassifier.from_pretrained(
                model_path=output_dir,
                num_labels=num_labels
            )
            self.epochs=1
        else:
            print(f"🆕 No saved model found. Initializing {model_name} from scratch...")
            self.model = CustomDebertaClassifier(model_name, num_labels=num_labels)

        self.trainer = None

    def train(self, train_csv, valid_csv=None):
        """Train or continue training from saved model."""
        print("🚀 Training started...")

        # Load datasets
        ds = load_csvs(train_csv, valid_csv)
        ds = ds.map(lambda ex, idx: {"id": idx, **ex}, with_indices=True)

        # Prepare datasets
        train_dataset = PhishDataset(
            ds["train"], self.tokenizer,
            max_length=self.max_length,
            stride=self.stride,
            mode="train"
        )

        eval_dataset = None
        if "validation" in ds:
            eval_dataset = PhishDataset(
                ds["validation"], self.tokenizer,
                max_length=self.max_length,
                stride=self.stride,
                mode="eval"
            )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="longest")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True if eval_dataset is not None else False,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none",
        )

        # ✅ Hugging Face Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
        )

        # Continue training
        self.trainer.train()

        # Save model and tokenizer
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✅ Model and tokenizer saved to {self.output_dir}")

        # Evaluate
        if eval_dataset:
            metrics = self.trainer.evaluate()
            print("📊 Validation metrics:", metrics)
