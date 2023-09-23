import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your maintenance dataset (replace 'data.csv' with your dataset file)
data = pd.read_csv('data.csv')

# Assuming you have features like temperature, pressure, vibration, etc., and a target variable 'label' indicating maintenance needed or not.
# Replace 'X' and 'y' with your actual feature columns and target variable.
X = data[['temperature', 'pressure', 'vibration']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier (you can use other algorithms as well)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Now, you can use the trained model for predictive maintenance.
# For example, you can predict maintenance needs for new data points like this:
new_data_point = [[75, 100, 3.5]]  # Replace with your actual data
predicted_maintenance = model.predict(new_data_point)

if predicted_maintenance[0] == 1:
    print("Maintenance needed.")
else:
    print("No maintenance needed.")
