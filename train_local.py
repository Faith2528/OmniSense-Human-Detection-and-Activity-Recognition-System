import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import DataLoader, TensorDataset

# --- 1. ARCHITECTURE (The 52x400 Blueprint) ---
class Rescue1DCNN(nn.Module):
    def __init__(self):
        super(Rescue1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(52, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4), # High dropout prevents it from memorizing the exact spots you fell
            nn.Linear(64, 3) 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- 2. LOAD & NORMALIZE ---
def prepare_data():
    X, y = [], []
    class_map = {'empty': 0, 'walk': 1, 'fall': 2}
    
    for label, idx in class_map.items():
        files = glob.glob(f"dataset/{label}_*.csv")
        if not files:
            print(f"⚠️ Warning: No files found for '{label}'!")
            continue
            
        print(f"📊 Loading {len(files)} files for {label}...")
        for f in files:
            try:
                data = pd.read_csv(f, header=None).values
                # Ensure the shape is exactly what we expect (400, 52)
                if data.shape != (400, 52):
                    continue 
                    
                # Normalize the signal to handle slight drops in WiFi strength
                data = (data - np.mean(data)) / (np.std(data) + 1e-8)
                X.append(data.T) # Transpose to (52, 400) for PyTorch
                y.append(idx)
            except Exception as e:
                pass # Skip corrupted files silently
                
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

# --- 3. THE TRAINING LOOP ---
def train_model():
    print("Checking dataset...")
    X, y = prepare_data()
    
    if len(X) == 0:
        print("❌ Error: No valid data loaded. Check your dataset folder.")
        return

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = Rescue1DCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n🧠 Training on {len(X)} total samples...")
    model.train()
    
    # 60 Epochs is the sweet spot for a dataset this size
    for epoch in range(60):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print progress every 10 epochs
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss/len(loader)
            print(f"Epoch [{epoch+1}/60] | Loss: {avg_loss:.4f}")

    # Save the compiled weights
    torch.save(model.state_dict(), "rescue_wifi_brain.pth")
    print("\n✅ DONE! 'rescue_wifi_brain.pth' has been generated.")

if __name__ == "__main__":
    train_model()