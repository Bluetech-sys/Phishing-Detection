import torch
import numpy as np
from Train_model import Trainer
from predict_text import Predict_phishing
from Text_Transformer import aggregate_chunk_predictions
import macros

def main():
    # -------------------------------
    # Step 1: Train or continue training
    # -------------------------------
    trainer = Trainer(
        model_name=macros.MODEL_NAME,
        num_labels=macros.NUM_LABELS,
        max_length=macros.MAX_LENGTH,
        stride=macros.STRIDE,
        epochs=macros.EPOCHS,
        batch_size=macros.BATCH_SIZE,
        lr=macros.LR,
        output_dir=macros.OUTPUT_DIR
    )

    # Train only if called
    train_choice = input("Do you want to train the model? (y/n): ").strip().lower()
    if train_choice == "y":
        train_csv = input("Enter path to training CSV: ").strip()
        valid_csv = input("Enter path to validation CSV (or leave blank): ").strip() or None
        trainer.train(train_csv, valid_csv)

    # -------------------------------
    # Step 2: Load model for prediction
    # -------------------------------
    predictor = Predict_phishing(model_path=macros.OUTPUT_DIR)

    # -------------------------------
    # Step 3: Predict sample text
    # -------------------------------
    orig_ids = []
    logits = []

    while True:
        text = input("\nEnter text to predict (or 'exit' to quit): ").strip()
        if text.lower() == "exit":
            break
    # Let's say predictor returns (orig_id, logits) for each text
        oid, logit = predictor.predict_text(text)  # adjust depending on your API
        orig_ids.append(oid)
        logits.append(logit)
        pred_label = int(torch.softmax(torch.from_numpy(logit), dim=-1).argmax())
        pred_confidence = float(torch.softmax(torch.from_numpy(logit), dim=-1).max())
        print(f"\nPrediction: {pred_label} (confidence: {pred_confidence:.4f})")

# After input ends, aggregate predictions by majority:
agg_results = aggregate_chunk_predictions(orig_ids, np.array(logits), method="majority")

print("\n=== Aggregated Majority Results ===")
for oid, label, conf in agg_results:
    print(f"ID: {oid}, Label: {label}, Confidence: {conf:.4f}")

if __name__ == "__main__":
    main()
