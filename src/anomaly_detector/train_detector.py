import os
import pandas as pd
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib
from sentence_transformers import SentenceTransformer

def parse_log_line(line):
    log_pattern = re.compile(r'(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+([\w.-]+)\s+([\w-]+)\[(\d+)\]:\s+(.*)')
    match = log_pattern.match(line)
    if match:
        _, hostname, process, pid, message = match.groups()
        return {'process': process, 'message': message.strip()}
    return None

def main():
    log_file_path = 'data/normal_traffic.log'
    model_dir = 'models'

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    print("🔹 Starting anomaly detector training with feature scaling...")

    # Load and parse logs
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    parsed_logs = [parse_log_line(line) for line in lines if parse_log_line(line)]
    df = pd.DataFrame(parsed_logs)

    if df.empty:
        raise ValueError("❌ No valid log entries found. Check your log file format.")

    print(f"✅ Parsed {len(df)} log entries")

    # Generate sentence embeddings
    print("🔹 Generating sentence embeddings...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    message_embeddings = embedding_model.encode(df['message'].tolist(), show_progress_bar=True)

    # Normalize embeddings
    message_features_df = pd.DataFrame(message_embeddings)

    # Structural and categorical features
    print("🔹 Extracting structural and categorical features...")
    df['msg_length'] = df['message'].str.len()
    df['special_chars'] = df['message'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))
    structural_features = df[['msg_length', 'special_chars']]
    process_features = pd.get_dummies(df['process'], prefix='proc')

    # Combine all features
    features = pd.concat([
        structural_features.reset_index(drop=True),
        message_features_df.reset_index(drop=True),
        process_features.reset_index(drop=True)
    ], axis=1)
    features.columns = features.columns.astype(str)

    print(f"✅ Created feature matrix with shape: {features.shape}")

    # Scale features
    print("🔹 Scaling features using RobustScaler...")
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)

    # Train Isolation Forest model
    print("🔹 Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(features_scaled)

    # Save model artifacts
    joblib.dump(model, os.path.join(model_dir, 'isolation_forest_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(features.columns.tolist(), os.path.join(model_dir, 'feature_columns.joblib'))

    print("\n✅ Model, scaler, and feature columns saved successfully in 'models/'.")

if __name__ == "__main__":
    main()
