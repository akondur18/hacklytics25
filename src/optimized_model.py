import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score 
from imblearn.over_sampling import SMOTE
import re
from imblearn.over_sampling import BorderlineSMOTE, ADASYN

# Set all random seeds
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom Dataset Class
class CancerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Enhanced Neural Network Architecture
class CancerClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 4)  
        )

    def forward(self, x):
        return self.network(x)

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.25, 0.75, 2.0, 1.5], gamma=2):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()

# Clean Histology labels (removes leading/trailing spaces and normalizes spaces in between)
def clean_histology_labels(df):
    df['Histology'] = df['Histology'].apply(lambda x: re.sub(r'\\s+', ' ', x.strip().title()))
    return df

# Enhanced Feature Engineering
def add_features(df):
    df = df.copy()
    df['Protein_Volatility'] = df[['Protein1', 'Protein2', 'Protein3', 'Protein4']].std(axis=1)
    df['Clinical_Risk_Score'] = (
        df['HER2 status'].map({'Negative': 0, 'Positive': 1}).fillna(0) * 2 +
        df['ER status'].map({'Negative': 0, 'Positive': 1}).fillna(0) +
        df['PR status'].map({'Negative': 0, 'Positive': 1}).fillna(0) +
        df['Tumour_Stage'].map({'I': 1, 'II': 2, 'III': 3}).fillna(1)
    )
    return df

# Load and merge datasets
def load_and_merge_data(original_path, synthetic_path):
    original_data = pd.read_csv(original_path).dropna()
    synthetic_data = pd.read_csv(synthetic_path).dropna()
    original_data['Histology'] = original_data['Histology'].str.strip()
    synthetic_data['Histology'] = synthetic_data['Histology'].str.strip()
    original_data['data_source'] = 'original'
    synthetic_data['data_source'] = 'synthetic'

    # Clean Histology labels in both datasets
    original_data = clean_histology_labels(original_data)
    synthetic_data = clean_histology_labels(synthetic_data)

    if set(original_data.columns) != set(synthetic_data.columns):
        raise ValueError("Column mismatch between original and synthetic datasets")

    combined_data = pd.concat([original_data, synthetic_data], ignore_index=True).dropna()

    #print("\nUnique Histology Classes:", combined_data['Histology'].unique())
    return combined_data

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005  # You'll likely tune this
PATIENCE = 25
SEED = 42
ALPHA = [1 / 0.7, 1 / 0.2, 1 / 0.1, 1 / 0.15] # Initial alpha values

if __name__ == "__main__":
    set_seeds(SEED)

    data = load_and_merge_data(
        original_path='/Users/anushkakondur/hacklytics25/BRCA.csv',
        synthetic_path='/Users/anushkakondur/hacklytics25/BRCA_synthetic_clean (1).csv'
    )

    data = add_features(data)

    train_data, test_data = train_test_split(
        data, test_size=0.2, stratify=data['Histology'], random_state=SEED
    )
    train_data['Histology'] = train_data['Histology'].str.strip()
    test_data['Histology'] = test_data['Histology'].str.strip()
    histology_encoder = LabelEncoder().fit(train_data['Histology'])
    num_classes = len(histology_encoder.classes_)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), [
            'Protein1', 'Protein2', 'Protein3', 'Protein4',
            'Protein_Volatility', 'Clinical_Risk_Score', 'Age'
        ]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), [
            'Tumour_Stage', 'HER2 status', 'PR status'
        ])
    ])
    X = preprocessor.fit_transform(data)  # Fit and transform all data
    y = histology_encoder.transform(data['Histology'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=data['Histology'], random_state=SEED
    )

    best_model_overall = None
    best_f1_overall = 0
    best_sampler_name = None

    smote_variants = {
        "SMOTE": SMOTE(k_neighbors=min(4, min(np.unique(y_train, return_counts=True)[1]) - 1), random_state=SEED),
        "BorderlineSMOTE": BorderlineSMOTE(k_neighbors=min(4, min(np.unique(y_train, return_counts=True)[1]) - 1), random_state=SEED),
        "ADASYN": ADASYN(n_neighbors=min(4, min(np.unique(y_train, return_counts=True)[1]) - 1), random_state=SEED),
    }

    for name, sampler in smote_variants.items():
        try:
            X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
            print(f"\nClass distribution after {name} resampling:")
            print(dict(zip(*np.unique(y_train_res, return_counts=True))))

            train_dataset = CancerDataset(X_train_res, y_train_res)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
            test_dataset = CancerDataset(X_test, y_test) 
            model = CancerClassifier(X_train_res.shape[1])
            criterion = FocalLoss(alpha=ALPHA)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=EPOCHS)

            best_val_f1, patience_counter = 0, 0

            for epoch in range(EPOCHS):
                model.train()
                for inputs, labels in train_loader:
                    alpha = 0.4  # Adjust alpha value for Mixup
                    if np.random.rand() < 0.5:  # Apply Mixup 50% of the time
                        lam = np.random.beta(alpha, alpha)
                        index = torch.randperm(inputs.size(0))
                        inputs_mixed = lam * inputs + (1 - lam) * inputs[index]
                        labels_mixed = lam * labels + (1 - lam) * labels[index]

                        # One-hot encode labels for Mixup
                        #labels_mixed = torch.nn.functional.one_hot(labels_mixed.long(), num_classes=num_classes).float()
                        
                        optimizer.zero_grad()
                        outputs = model(inputs_mixed)

                        loss = nn.CrossEntropyLoss()(outputs, labels_mixed.long())
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                    else:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)  
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                model.eval()
                with torch.no_grad():
                    aug_preds = [
                        model(test_dataset.features + torch.randn_like(test_dataset.features) * 0.05).argmax(1)
                        for _ in range(5) 
                    ]
                    val_preds = torch.mode(torch.stack(aug_preds), dim=0)[0]
                    val_f1 = f1_score(y_test, val_preds, average='macro')

                    #print(f"Epoch {epoch + 1}: Macro F1 = {val_f1:.2f}")

                    if val_f1 > best_val_f1:
                        best_val_f1, patience_counter = val_f1, 0
                        torch.save(model.state_dict(), 'best_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            #print(f"Early stopping at epoch {epoch + 1}")
                            break

            # Load the best model for this sampler
            model.load_state_dict(torch.load('best_model.pth'))

            if best_val_f1 > best_f1_overall:
              best_f1_overall = best_val_f1
              best_model_overall = model
              best_sampler_name = name


        except ValueError as e:
            print(f"{name} failed: {e}")
            continue
        except RuntimeError as e: # Catch potential runtime errors too
            print(f"{name} training failed: {e}")
            continue
    # Evaluate the overall best model
    if best_model_overall:
        best_model_overall.eval()  # Set to eval mode
        with torch.no_grad():  # Disable gradients
            final_preds = torch.mode(torch.stack([
                best_model_overall(test_dataset.features + torch.randn_like(test_dataset.features) * 0.05).argmax(1)
                for _ in range(7)  # Number of augmentations during testing
            ]), dim=0)[0]

        results = test_data.copy()
        results['Predicted'] = histology_encoder.inverse_transform(final_preds.numpy())
        results['Actual'] = histology_encoder.inverse_transform(y_test)

        print("\nOverall Classification Report (Best Model):")
        print(classification_report(y_test, final_preds.numpy(), target_names=histology_encoder.classes_))

        print("\nSample Predictions vs Actual (Best Model):")
        print(results[['data_source', 'Protein_Volatility', 'Clinical_Risk_Score', 'Predicted', 'Actual']].sample(10, random_state=SEED).to_string(index=False))

        accuracy = accuracy_score(y_test, final_preds.numpy())
        print(f"\nAccuracy Score (Best Model): {accuracy:.4f}")
        print(f"Best Sampler: {best_sampler_name}")
    else:
        print("No suitable model found after trying all samplers.")
