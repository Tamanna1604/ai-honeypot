import pandas as pd
import re
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from sentence_transformers import SentenceTransformer

def parse_log_line(line):
    """Parse a single log line into structured fields."""
    log_pattern = re.compile(r'(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+([\w-]+)\s+([\w]+)\[(\d+)\]:\s+(.*)')
    match = log_pattern.match(line)
    if match:
        timestamp_str, hostname, process, pid, message = match.groups()
        return {'process': process, 'message': message.strip()}
    return None


def main():
    log_file_path = 'data/normal_traffic.log'
    print("Starting anomaly detector training with scaling...")

    # --- Load and parse logs ---
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    parsed_logs = [parse_log_line(line) for line in lines if parse_log_line(line)]
    df = pd.DataFrame(parsed_logs)

    print("Performing feature engineering...")

    # --- Encode message embeddings ---
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    message_embeddings = embedding_model.encode(df['message'].tolist(), show_progress_bar=True)
    message_features_df = pd.DataFrame(message_embeddings)

    # --- Create structural and categorical features ---
    df['msg_length'] = df['message'].str.len()
    df['special_chars'] = df['message'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))
    structural_features = df[['msg_length', 'special_chars']]
    process_features = pd.get_dummies(df['process'], prefix='proc')

    # --- Combine all features ---
    features = pd.concat([
        structural_features.reset_index(drop=True),
        message_features_df.reset_index(drop=True),
        process_features.reset_index(drop=True)
    ], axis=1)
    features.columns = features.columns.astype(str)

    # --- Scale features ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    print(f"Created scaled feature matrix with shape: {X_scaled.shape}")

    # --- Train Isolation Forest ---
    print("Training the Isolation Forest model...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,  # fine-tuned contamination
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # --- Compute anomaly scores and adaptive threshold ---
    scores = model.decision_function(X_scaled)
    threshold = np.percentile(scores, 2)  # top 2% anomalies
    print(f"[INFO] Adaptive threshold set to: {threshold:.4f}")

    # --- Save all model artifacts ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/isolation_forest_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(features.columns.tolist(), 'models/feature_columns.joblib')
    np.save('models/if_threshold.npy', threshold)

    print("\n✅ Model, scaler, feature columns, and threshold saved successfully.")


if __name__ == "__main__":
    main()
