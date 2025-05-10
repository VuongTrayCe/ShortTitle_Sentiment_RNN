import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from sklearn.metrics import accuracy_score, f1_score
from data import train_loader, test_loader
from model import model_glove, model_scratch
import pandas as pd

def train_and_evaluate_model(model, train_loader, test_loader, model_name="model", num_epochs=100, learning_rate=0.01, save_path="metrics.json"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training {model_name} on {device} ---")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels_all = []

        for batch_sequences, batch_labels in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_sequences)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            # total_train_loss += loss.item()
            preds_class = torch.argmax(predictions, dim=1)
            train_preds.extend(preds_class.cpu().numpy())
            train_labels_all.extend(batch_labels.cpu().numpy())

        # avg_epoch_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels_all, train_preds)
        train_f1 = f1_score(train_labels_all, train_preds, average='macro')

        # --- Validation ---
        model.eval()
        val_preds = []
        val_labels_all = []
        total_val_loss = 0

        with torch.no_grad():
            for val_sequences, val_labels in test_loader:
                val_sequences = val_sequences.to(device)
                val_labels = val_labels.to(device)

                val_predictions = model(val_sequences)
                loss = criterion(val_predictions, val_labels)
                total_val_loss += loss.item()

                preds_class = torch.argmax(val_predictions, dim=1)
                val_preds.extend(preds_class.cpu().numpy())
                val_labels_all.extend(val_labels.cpu().numpy())

        # avg_val_loss = total_val_loss / len(test_loader)
        val_accuracy = accuracy_score(val_labels_all, val_preds)
        val_f1 = f1_score(val_labels_all, val_preds, average='macro')

        # Print mỗi 10 epoch hoặc epoch đầu tiên
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.4f}, Train F1-score: {train_f1:.4f} ---- Val Acc: {val_accuracy:.4f}, Val F1-score: {val_f1:.4f}')

    # --- Load existing metrics (if any) ---
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    # --- Save only final val metrics ---
    metrics[model_name] = {
        "val_accuracy": round(val_accuracy, 4),
        "val_f1_score": round(val_f1, 4)
    }

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Saved metrics for {model_name} to {save_path}")
    
print("\n--------------------------------Training----------------------------------")

for (model,model_name) in [(model_glove,"Model_pretrained"),(model_scratch,"Model_scratch")]:
    train_and_evaluate_model(model,train_loader,test_loader,model_name=model_name,num_epochs=50)
print("\n--------------------------------Result----------------------------------\n")
# Đọc file JSON
with open("metrics.json", "r") as f:
    data = json.load(f)

# Tạo DataFrame
df = pd.DataFrame.from_dict(data, orient='index')
print(df)