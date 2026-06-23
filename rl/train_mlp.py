import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# === 1. Загрузка данных с автоматическим определением разделителя ===
print("📊 Loading dataset...")

# Пробуем разные разделители
csv_path = './synthetic_guidance_data.csv'
for sep in ['\t', ',', r'\s+']:
    try:
        df = pd.read_csv(csv_path, sep=sep, engine='python')
        if len(df.columns) >= 12:  # У нас 9 входов + 3 выхода = 12 столбцов
            print(f"   ✓ Detected separator: {repr(sep)}")
            print(f"   Columns: {list(df.columns)}")
            break
    except Exception as e:
        continue
else:
    print("❌ Could not detect separator. Trying without header...")
    # Если не получилось, читаем без заголовка
    df = pd.read_csv(csv_path, sep=r'\s+', header=None, engine='python')
    # Назначаем имена столбцов вручную
    df.columns = ['u_los_x', 'u_los_y', 'u_los_z', 
                  'lambda_x', 'lambda_y', 'lambda_z',
                  'pitch', 'yaw', 'roll',
                  'acc_x', 'acc_y', 'acc_z']

print(f"   Loaded {len(df)} samples")
print(f"   Columns: {list(df.columns)}")

# Удаление NaN
df = df.dropna()
print(f"   After NaN removal: {len(df)} samples")

# Разделение на X и y
X = df[['u_los_x', 'u_los_y', 'u_los_z', 
        'lambda_x', 'lambda_y', 'lambda_z',
        'pitch', 'yaw', 'roll']].values.astype(np.float32)

y = df[['acc_x', 'acc_y', 'acc_z']].values.astype(np.float32)

print(f"   Input shape: {X.shape}, Output shape: {y.shape}")
print(f"   Output range: acc=[{y.min():.3f}, {y.max():.3f}]")

# === 2. Нормализация ===
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

y_max = np.maximum(np.abs(y.min(axis=0)), np.abs(y.max(axis=0)))
y_max = np.maximum(y_max, 1e-6)
y_scaled = y / y_max

print(f"   y_max (for denormalization): {y_max}")

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=0.15, random_state=42
)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)

# === 3. Архитектура MLP ===
class GuidanceMLP(nn.Module):
    def __init__(self, input_dim=9, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

# === 4. Обучение ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🧠 Training on: {device}")

model = GuidanceMLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ✅ ИСПРАВЛЕНИЕ: убран аргумент verbose=True
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
criterion = nn.MSELoss()

best_val_loss = float('inf')
patience = 15
patience_counter = 0

print("\n🚀 Starting training...")
for epoch in range(200):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        # Выводим прогресс вручную вместо verbose
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_guidance_mlp.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break
            
            
# === 5. Экспорт в ONNX ===
print("\n📦 Exporting to ONNX...")
model.load_state_dict(torch.load('best_guidance_mlp.pth', map_location=device))
model.eval()

dummy_input = torch.randn(1, 9).to(device)
onnx_path = './guidance_mlp.onnx'
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

torch.onnx.export(
    model, dummy_input, onnx_path,
    export_params=True,  
    input_names=['state'],
    output_names=['action'],
    dynamic_axes={'state': {0: 'batch'}, 'action': {0: 'batch'}},
    opset_version=13,
    do_constant_folding=True,
    dynamo=False 
)
print(f"✅ Model saved to: {onnx_path}")

# === 6. Сохранение параметров нормализации ===
scaler_path = './scaler_params.npz'
np.savez(
    scaler_path,
    X_mean=scaler_X.mean_.astype(np.float32),
    X_scale=scaler_X.scale_.astype(np.float32),
    y_max=y_max.astype(np.float32)
)
print(f"✅ Scaler params saved to: {scaler_path}")

# Сохранение в текстовом формате для C++
txt_path = './scaler_params.txt'
with open(txt_path, 'w') as f:
    f.write(' '.join(map(str, scaler_X.mean_)) + '\n')
    f.write(' '.join(map(str, scaler_X.scale_)) + '\n')
    f.write(' '.join(map(str, y_max)) + '\n')
print(f"✅ Scaler params (text) saved to: {txt_path}")

# === 7. Проверка качества ===
print("\n📈 Quality check...")
model.eval()
with torch.no_grad():
    X_val_t = X_val_t.to(device)
    y_val_t = y_val_t.to(device)
    pred = model(X_val_t)
    mse = criterion(pred, y_val_t).item()
    
    pred_real = pred.cpu().numpy() * y_max
    y_real = y_val_t.cpu().numpy() * y_max
    mae = np.mean(np.abs(pred_real - y_real))
    
    print(f"   Validation MSE (normalized): {mse:.6f}")
    print(f"   Validation MAE (real units): {mae:.4f} m/s²")
    print(f"   Max absolute error: {np.max(np.abs(pred_real - y_real)):.4f} m/s²")

print("\n🎉 Done! Next step: integrate into C++ via ONNX Runtime")
