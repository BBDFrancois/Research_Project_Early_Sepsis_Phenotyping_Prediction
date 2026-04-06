import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

from pypots.imputation import SAITS
from pypots.representation import TS2Vec
from processor import SepsisDataProcessor


class SimpleAutoEncoder(nn.Module):
    """
    Autoencoder for dimensionality reduction of flattened temporal embeddings.
    Encoder compresses input to a latent representation, decoder reconstructs it.
    """

    def __init__(self, input_dim, latent_dim=32):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class SepsisEncodingPipeline:
    """
    End-to-end encoding pipeline for the sepsis dataset.
    Sequentially applies SAITS imputation, TS2Vec temporal embedding,
    and an autoencoder to produce fixed-size patient-level representations.
    Pre-trained models can be injected to skip individual training steps.
    """

    def __init__(self,
                 file_path='data/Dataset.csv',
                 window_start=-17,
                 window_end=-6,
                 min_presence_pct=0.5,
                 essential_cols=None,
                 validation_split=0.2,
                 saits_epochs=10,
                 saits_batch_size=32,
                 ts2vec_output_dims=32,
                 ts2vec_epochs=20,
                 ts2vec_batch_size=32,
                 ae_latent_dim=32,
                 ae_epochs=40,
                 ae_batch_size=64,
                 ae_lr=1e-3,
                 device=None,
                 random_state=42):

        self.file_path = file_path
        self.essential_cols = essential_cols
        self.validation_split = validation_split
        self.random_state = random_state

        self.window_params = {
            'window_start': window_start, 'window_end': window_end,
            'start_for_non_sepsis': True, 'non_sepsis_ratio': 1.0
        }
        self.min_presence_pct = min_presence_pct

        self.saits_params = {
            'n_layers': 2, 'd_model': 128, 'n_heads': 4, 'd_k': 32, 'd_v': 32,
            'd_ffn': 128, 'dropout': 0.1, 'epochs': saits_epochs, 'batch_size': saits_batch_size
        }
        self.ts2vec_params = {
            'n_output_dims': ts2vec_output_dims, 'd_hidden': 32, 'n_layers': 2,
            'mask_mode': 'binomial', 'epochs': ts2vec_epochs, 'batch_size': ts2vec_batch_size
        }
        self.ae_params = {
            'latent_dim': ae_latent_dim, 'epochs': ae_epochs,
            'batch_size': ae_batch_size, 'lr': ae_lr
        }

        if device is None:
            import torch_directml
            if torch_directml.is_available():
                self.device = torch_directml.device()
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.scaler_ae = StandardScaler()

    @staticmethod
    def save_model(model, path):
        """Saves a PyTorch model to the specified path, creating directories as needed."""
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model, path)
            print(f"Model saved: {path}")
        except Exception as e:
            print(f"Error saving {path}: {e}")

    def _plot_training_curve(self, train_history, val_history, title="Training Curve"):
        """Plots train and validation loss curves over epochs."""
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(train_history, label='Train Loss', color='blue')
            if val_history:
                plt.plot(val_history, label='Validation Loss', color='orange', linestyle='--')
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Could not display plot ({e})")

    def run(self, saits_model=None, ts2vec_model=None, ae_model=None):
        """
        Executes the full encoding pipeline.
        If a model is provided for any step, training is skipped and inference is run directly.
        Returns encoded train/test arrays, labels, patient IDs, and the three trained models.
        """

        # Step 1: Data preparation
        print("Step 1: Data preparation and train/test split")
        processor = SepsisDataProcessor(self.file_path)
        processor.initial_prep()
        processor.window_selection(**self.window_params)
        processor.filter_physio_variables()

        df_cleaned = processor.filter_variables(
            min_patient_presence_pct=self.min_presence_pct,
            essential_cols=self.essential_cols
        )
        print(df_cleaned.info())

        df_train_global, df_test = processor.split_train_test()

        has_test_data = False
        X_test_pids = np.array([])
        y_test = np.array([])
        X_test_encoded = np.array([])

        if df_test is not None and not df_test.empty:
            has_test_data = True
            X_test_pids = df_test.sort_values(['Patient_ID', 'Hour'])['Patient_ID'].unique()
            X_test_raw, y_test = processor.to_tensor(df_test)
        else:
            print("Warning: test set is empty, test processing will be skipped.")
            X_test_raw = None

        X_train_pids = df_train_global.sort_values(['Patient_ID', 'Hour'])['Patient_ID'].unique()
        X_train_global, y_train = processor.to_tensor(df_train_global)

        if has_test_data:
            X_train_global, X_test_raw = processor.standardize_tensors(X_train_global, X_test_raw)
        else:
            dummy = X_train_global.copy()
            X_train_global, _ = processor.standardize_tensors(X_train_global, dummy)

        n_steps, n_features = X_train_global.shape[1], X_train_global.shape[2]

        # Step 2: SAITS imputation
        print("Step 2: SAITS imputation")
        if saits_model is None:
            print("Training SAITS...")
            saits = SAITS(n_steps=n_steps, n_features=n_features, device=self.device,
                          saving_path="models/saits_tmp", **self.saits_params)
            saits.fit(train_set={"X": X_train_global})
            saits_model = saits
        else:
            print("Pre-trained SAITS provided, running inference.")

        X_train_imp = saits_model.predict({"X": X_train_global})["imputation"].astype('float32')
        if has_test_data:
            X_test_imp = saits_model.predict({"X": X_test_raw})["imputation"].astype('float32')
        else:
            X_test_imp = None

        # Step 3: TS2Vec temporal embedding
        print("Step 3: TS2Vec embedding")
        if ts2vec_model is None:
            print("Training TS2Vec...")
            ts2vec = TS2Vec(n_steps=n_steps, n_features=n_features, device=self.device,
                            saving_path="models/ts2vec_tmp", **self.ts2vec_params)
            ts2vec.fit(train_set={"X": X_train_imp})
            ts2vec_model = ts2vec
        else:
            print("Pre-trained TS2Vec provided, running inference.")

        emb_train = ts2vec_model.predict({"X": X_train_imp})["representation"]
        N_train, T, F_emb = emb_train.shape
        X_train_flat = emb_train.reshape(N_train, T * F_emb)

        if has_test_data and X_test_imp is not None:
            emb_test = ts2vec_model.predict({"X": X_test_imp})["representation"]
            X_test_flat = emb_test.reshape(emb_test.shape[0], T * F_emb)
        else:
            X_test_flat = None

        # Step 4: Autoencoder
        print("Step 4: Autoencoder")
        X_train_ae_in = self.scaler_ae.fit_transform(X_train_flat)
        tensor_train_global = torch.FloatTensor(X_train_ae_in).to(self.device)

        if has_test_data and X_test_flat is not None:
            X_test_ae_in = self.scaler_ae.transform(X_test_flat)
            tensor_test = torch.FloatTensor(X_test_ae_in).to(self.device)
        else:
            tensor_test = None

        if ae_model is None:
            print("Training Autoencoder...")
            idx = np.arange(len(X_train_global))
            train_idx, val_idx = train_test_split(idx, test_size=self.validation_split, random_state=self.random_state)

            X_train_ae_sub = X_train_ae_in[train_idx]
            X_val_ae = X_train_ae_in[val_idx]

            tensor_train_sub = torch.FloatTensor(X_train_ae_sub).to(self.device)
            tensor_val = torch.FloatTensor(X_val_ae).to(self.device)

            ae = SimpleAutoEncoder(X_train_flat.shape[1], self.ae_params['latent_dim']).to(self.device)
            optimizer = optim.Adam(ae.parameters(), lr=self.ae_params['lr'])
            criterion = nn.MSELoss()

            train_loader = DataLoader(TensorDataset(tensor_train_sub), batch_size=self.ae_params['batch_size'],
                                      shuffle=True)
            val_loader = DataLoader(TensorDataset(tensor_val), batch_size=self.ae_params['batch_size'], shuffle=False)

            history_train, history_val = [], []

            for epoch in range(self.ae_params['epochs']):
                ae.train()
                t_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    _, decoded = ae(batch[0])
                    loss = criterion(decoded, batch[0])
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item()
                history_train.append(t_loss / len(train_loader))

                ae.eval()
                v_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        _, decoded = ae(batch[0])
                        v_loss += criterion(decoded, batch[0]).item()
                history_val.append(v_loss / len(val_loader))

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1:02d} | Train: {history_train[-1]:.4f} | Val: {history_val[-1]:.4f}")

            self._plot_training_curve(history_train, history_val, title="Autoencoder Training")
            ae_model = ae
        else:
            print("Pre-trained Autoencoder provided, running inference.")
            ae_model = ae_model.to(self.device)

        # Step 5: Final inference
        ae_model.eval()
        with torch.no_grad():
            encoded_train, _ = ae_model(tensor_train_global)
            X_train_encoded = encoded_train.cpu().numpy()

            if tensor_test is not None:
                encoded_test, _ = ae_model(tensor_test)
                X_test_encoded = encoded_test.cpu().numpy()
            else:
                X_test_encoded = np.array([])

        print(f"Pipeline complete. Train shape: {X_train_encoded.shape}, Test shape: {X_test_encoded.shape}")

        return (X_train_encoded, X_test_encoded, y_train, y_test, X_train_pids, X_test_pids), (
            saits_model, ts2vec_model, ae_model)

    @staticmethod
    def save_processed_data(data_out, data_centered_out, folder="../data"):
        """
        Saves both standard and centered encoded datasets into a single compressed .npz file.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        X_train, X_test, y_train, y_test, pid_train, pid_test = data_out
        X_cen_train, X_cen_test, y_cen_train, y_cen_test, pid_cen_train, pid_cen_test = data_centered_out

        data_dict = {
            'X_train_encoded': X_train,
            'X_test_encoded': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_pids': pid_train,
            'X_test_pids': pid_test,
            'X_centered_train_encoded': X_cen_train,
            'X_centered_test_encoded': X_cen_test,
            'y_centered_train': y_cen_train,
            'y_centered_test': y_cen_test,
            'X_centered_train_pids': pid_cen_train,
            'X_centered_test_pids': pid_cen_test
        }

        save_path = os.path.join(folder, "sepsis_processed_full.npz")
        np.savez_compressed(save_path, **data_dict)
        print(f"Data saved to: {save_path}")

    @staticmethod
    def load_processed_data(filepath="../data/sepsis_processed_full.npz"):
        """
        Loads encoded datasets from a previously saved .npz file.
        Returns two tuples: (data_out, data_centered_out).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        loaded = np.load(filepath, allow_pickle=True)

        print("Data loaded.")
        print(f"Available arrays: {loaded.files}")

        data_out = (
            loaded['X_train_encoded'], loaded['X_test_encoded'],
            loaded['y_train'], loaded['y_test'],
            loaded['X_train_pids'], loaded['X_test_pids']
        )

        data_centered_out = (
            loaded['X_centered_train_encoded'], loaded['X_centered_test_encoded'],
            loaded['y_centered_train'], loaded['y_centered_test'],
            loaded['X_centered_train_pids'], loaded['X_centered_test_pids']
        )

        return data_out, data_centered_out
