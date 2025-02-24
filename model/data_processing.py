import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset with explicit dtype handling
data_path = "data/hotel_bookings.csv"
df = pd.read_csv(data_path, low_memory=False)  # Avoid mixed type warnings

# Identify mixed-type columns and fix them
for col in df.columns:
    df[col] = df[col].astype(str)  # Convert all to string to prevent errors

# Drop irrelevant columns
drop_cols = ['reservation_status', 'reservation_status_date', 'arrival_date_year']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Convert categorical columns to numerical using Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features & target
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✅ Data preprocessing complete.")
