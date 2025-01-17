
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")
        validate_model(model, val_loader, criterion)

def validate_model(model, val_loader, criterion, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > threshold
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append((masks > 0.5).cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    precision = precision_score(all_targets, all_preds, zero_division=1)
    recall = recall_score(all_targets, all_preds, zero_division=1)
    f1 = f1_score(all_targets, all_preds, zero_division=1)
    accuracy = accuracy_score(all_targets, all_preds)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
