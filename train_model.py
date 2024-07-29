import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load and preprocess data
df = pd.read_csv('HR_dataset.csv')
target = df.pop('Salary')

# Convert categorical features to numeric
char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
label_mapping = {c: pd.factorize(df[c])[1] for c in char_cols}
df[char_cols] = df[char_cols].apply(lambda x: pd.factorize(x)[0])

# Scale data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Split data
x_train, x_test, y_train, y_test = train_test_split(df_scaled, target, test_size=0.2, random_state=42)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)

# Save the model and scaler
joblib.dump(clf, 'salary_model.pkl')
joblib.dump(scaler, 'salary_scaler.pkl')

print("Model training complete and saved as 'salary_model.pkl' and 'salary_scaler.pkl'")
