import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
data = pd.read_csv("bmi.csv") 

# 2. Preprocess Data
data = data.dropna()
if 'Gender' in data.columns:
    data = data.drop(['Gender'], axis=1) 

x = data.drop('Index', axis=1) # Features: Height, Weight
y = data['Index']              # Target: Index

# 3. Scale the features
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# 4. Split the data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=27)

# 5. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=27)
model.fit(x_train, y_train)

# 6. Save the Model and Scaler
joblib.dump(model, 'bmi_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved successfully! You can now start the backend.")