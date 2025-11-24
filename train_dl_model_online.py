import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

# -----------------------------
# Load dataset from UCI
# -----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ["mpg","cylinders","displacement","horsepower","weight",
           "acceleration","model_year","origin","car_name"]
df = pd.read_csv(url, delim_whitespace=True, names=columns, na_values='?')
df = df.dropna().reset_index(drop=True)

# -----------------------------
# Add India data
# -----------------------------
india_data = pd.DataFrame({
    "mpg": [25, 22, 28],
    "cylinders": [4, 4, 4],
    "displacement": [120, 130, 110],
    "horsepower": [80, 85, 70],
    "weight": [2500, 2400, 2300],
    "acceleration": [15, 16, 14],
    "model_year": [80, 81, 82],
    "origin": [4, 4, 4],
    "car_name": ["IndiaCar1","IndiaCar2","IndiaCar3"]
})
df = pd.concat([df, india_data], ignore_index=True)

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop(["mpg","car_name"], axis=1)
y = df["mpg"]

numeric_features = ["cylinders","displacement","horsepower","weight","acceleration","model_year"]
categorical_features = ["origin"]

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")  # use sparse_output for latest sklearn

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# -----------------------------
# Build DL model
# -----------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# -----------------------------
# Train model
# -----------------------------
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16)

# -----------------------------
# Save model & preprocessor
# -----------------------------
model.save("auto_mpg_dl_model_extended.h5")
joblib.dump(preprocessor, "preprocessor_extended.joblib")

print("✅ Training complete. Model and preprocessor saved!")
