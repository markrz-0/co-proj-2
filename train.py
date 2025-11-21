import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. Define the Neural Network Architecture
# ==========================================
class SimpleNN(nn.Module):
    def __init__(self, input_size=7, hidden_size=20, output_size=1):
        super(SimpleNN, self).__init__()
        # Input Layer -> Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Hidden Layer -> Output Layer
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# ==========================================
# 2. Data Loading and Processing
# ==========================================
def load_data(features_path, output_path):
    print(f"Loading data from {features_path}...")
    
    # Check if files exist
    if not os.path.exists(features_path) or not os.path.exists(output_path):
        raise FileNotFoundError(f"Files not found: {features_path} or {output_path}")

    # Load Features
    with open(features_path, 'r') as f:
        # varied splitting handles space, tab, or comma separation
        x_data = [[float(num) for num in line.strip().replace(',', ' ').split()] for line in f if line.strip()]

    # Load Labels/Output
    with open(output_path, 'r') as f:
        y_data = [[float(num) for num in line.strip().replace(',', ' ').split()] for line in f if line.strip()]

    # Convert to PyTorch Tensors
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    return x_tensor, y_tensor

# ==========================================
# 3. Main Execution Block
# ==========================================
if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 512
    LEARNING_RATE = 0.002
    EPOCHS = 200
    TRAIN_SPLIT = 0.8  # 80% training, 20% validation
    
    # --- DATASET CONFIGURATION ---
    # Add more file pairs here in the future
    DATA_SOURCES = [
        {
            'features': os.path.join('data', 'features_pack_1763733893.txt'), 
            'output': os.path.join('data', 'output_pack_1763733893.txt')
        },
        {
            'features': os.path.join('data', 'features_pack_1763736842.txt'), 
            'output': os.path.join('data', 'output_pack_1763736842.txt')
        },
        # Example of adding a future file:
        # {'features': 'data/features3.txt', 'output': 'data/output3.txt'},
    ]

    try:
        train_subsets = []
        val_subsets = []

        # 1. Load and Split each dataset individually
        for source in DATA_SOURCES:
            try:
                X, y = load_data(source['features'], source['output'])
                
                full_dataset = TensorDataset(X, y)
                total_count = len(full_dataset)
                train_count = int(TRAIN_SPLIT * total_count)
                val_count = total_count - train_count

                # Split this specific file 80/20
                train_ds, val_ds = random_split(full_dataset, [train_count, val_count])
                
                train_subsets.append(train_ds)
                val_subsets.append(val_ds)
                print(f" -> Added {train_count} train and {val_count} val samples from {source['features']}")

            except FileNotFoundError as e:
                print(f"Warning: {e}. Skipping this file pair.")

        if not train_subsets:
            raise ValueError("No valid datasets were loaded. Please check your files.")

        # 2. Combine all subsets into Master Datasets
        master_train_dataset = ConcatDataset(train_subsets)
        master_val_dataset = ConcatDataset(val_subsets)

        print(f"\nTotal Training Samples: {len(master_train_dataset)}")
        print(f"Total Validation Samples: {len(master_val_dataset)}")

        # 3. Create DataLoaders
        train_loader = DataLoader(master_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(master_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # --- DEVICE CONFIGURATION ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing Device: {device}")
        if device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

        # 4. Initialize Model, Loss, and Optimizer
        model = SimpleNN().to(device) # Move model to GPU if available
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Lists to store loss history for plotting
        train_loss_history = []
        val_loss_history = []

        print(f"\nStarting training...")
        print("-" * 60)

        # 5. Training Loop
        for epoch in range(EPOCHS):
            model.train()  # Set model to training mode
            running_loss = 0.0
            
            for i, (inputs, targets) in enumerate(train_loader):
                # Move data to the same device as the model
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # 6. Validation Step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to the same device as the model
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)

            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        print("-" * 60)
        print("Training Complete.")

        # 7. Plotting Results
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('loss_plot.png')
        print("Loss plot saved to 'loss_plot.png'")

        # 8. Save the Model
        model_path = 'simple_nn_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        plt.show()

    except Exception as e:
        print(f"\nAn error occurred: {e}")