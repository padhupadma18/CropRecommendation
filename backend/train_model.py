import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
data = pd.read_csv(r"C:\Users\shiva\Croprecommender\data\Crop_recommendation.csv")

   # adjust path if needed

# 2. Split features (X) and target (y)
X = data.drop("label", axis=1)   # all columns except label
y = data["label"]                # target column

# 3. Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Train model
model.fit(X_train, y_train)

# 6. Evaluate model
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully with accuracy: {accuracy:.2f}")

# 7. Save trained model
joblib.dump(model, "crop_recommendation.pkl")
print("💾 Model saved as crop_recommendation.pkl")


