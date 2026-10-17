import torch
import torch.nn as nn
import pandas as pd
import os

# Veri yolu
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
csv_path = os.path.join(data_dir, 'initial_doe.csv')

class SurrogateDNN(nn.Module):
    def __init__(self):
        super(SurrogateDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train():
    df = pd.read_csv(csv_path)
    X = torch.tensor(df[['Mach', 'AoA', 'Dynamic_Pressure', 'Stiffness']].values, dtype=torch.float32)
    y = torch.tensor(df['Flutter_Load'].values, dtype=torch.float32).view(-1, 1)

    model = SurrogateDNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("DNN Eğitimi başlıyor...")
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')

    # Modeli kaydet
    model_path = os.path.join(os.path.dirname(__file__), 'dnn_model.pth')
    torch.save(model.state_dict(), model_path)
    print("DNN Modeli başarıyla eğitildi ve kaydedildi.")

if __name__ == '__main__':
    train()