import os
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio.PDB import PDBParser

# === Step 1: Extract pLDDT from .pdb or .pdb.gz ===


def extract_plddt_scores(file_path, max_len=300):
    parser = PDBParser(QUIET=True)

    # Open PDB or .gz file
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as handle:
            structure = parser.get_structure("protein", handle)
    else:
        structure = parser.get_structure("protein", file_path)

    scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    scores.append(residue['CA'].bfactor)

    # Pad or truncate
    scores = scores[:max_len]
    if len(scores) < max_len:
        scores += [0.0] * (max_len - len(scores))
    return np.array(scores, dtype=np.float32)

# === Step 2: Autoencoder ===


class FoldAutoencoder(nn.Module):
    def __init__(self, input_dim=300, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# === Step 3: Training ===
def train_autoencoder(pdb_dir, epochs=20, lr=1e-3, save_path="trained_autoencoder.pt"):
    model = FoldAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    pdb_files = [
        os.path.join(pdb_dir, f)
        for f in os.listdir(pdb_dir)
        if f.endswith(".pdb") or f.endswith(".pdb.gz")
    ]
    data = [extract_plddt_scores(f) for f in pdb_files]

    for epoch in range(epochs):
        total_loss = 0
        for x in data:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            output = model(x_tensor)
            loss = loss_fn(output, x_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data):.6f}")

    # 🧠 Save the trained model to disk
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

    return model


# === Step 4: Inference ===
def infer_misfold(model, input_vector, threshold=0.01):
    model.eval()
    x = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        error = torch.mean((out - x) ** 2).item()
        return error > threshold, error


# === Main: Example Usage ===
if __name__ == "__main__":
    pdb_folder = "alphafold_pdbs"  # Folder with .pdb or .pdb.gz files
    model = train_autoencoder(pdb_folder, epochs=10)

    # Run inference on a test file
    test_file = os.path.join(
        pdb_folder, "AF-Q14204-F10-model_v4.pdb.gz")  # Example
    if os.path.exists(test_file):
        plddt = extract_plddt_scores(test_file)
        is_misfold, err = infer_misfold(model, plddt)
        print(f"\nProtein: {test_file}")
        print(f"Reconstruction error: {err:.4f}")
        print("Likely misfolded!" if is_misfold else "Structure seems typical.")
    else:
        print(f"File not found: {test_file}")

if __name__ == "__main__":
    pdb_folder = "alphafold_pdbs"
    model = train_autoencoder(pdb_folder, epochs=10, save_path="trained_autoencoder.pt")

    # Optional: Test one inference as before
    test_file = os.path.join(
        pdb_folder, "AF-Q14204-F10-model_v4.pdb.gz")  # Example
    if os.path.exists(test_file):
        plddt = extract_plddt_scores(test_file)
        is_misfold, err = infer_misfold(model, plddt)
        print(f"\nTest File: {test_file}")
        print(f"Reconstruction Error: {err:.4f}")
        print("Likely Misfolded!" if is_misfold else "Structure seems typical.")
