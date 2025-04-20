import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the datasets
df1 = pd.read_csv('Datasets/car_price_data1.csv')
df2 = pd.read_csv('Datasets/car_price_data2.csv')

# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)

# Check dataset shape before removing missing values
print("Before dropna, dataset shape:", df.shape)

# Drop rows with missing values
df.dropna(subset=['Fuel_Type', 'Seller_Type', 'Transmission', 'Selling_Price', 'Present_Price'], inplace=True)

# Check dataset shape after cleaning
print("After dropna, dataset shape:", df.shape)

# Calculate car age from year
df['Age'] = 2025 - df['Year']

# Encode categorical columns
label_encoders = {}
for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df[['Present_Price', 'Age', 'Fuel_Type', 'Seller_Type', 'Transmission']]
y = df['Selling_Price']

# Ensure there's enough data to split
if len(df) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as 'model.pkl'")
else:
    print("Error: No valid data available for training.")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
