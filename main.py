# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd

df = pd.read_csv('/content/parkinsons.csv')
display(df.head())

X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

print("Input features (X):")
display(X.head())
print("\nOutput feature (y):")
display(y.head())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print("Scaled Input features (X):")
display(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5) # Using 5 neighbors as a common starting point
model.fit(X_train, y_train)

print("K-Nearest Neighbors model trained successfully!")

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

if accuracy >= 0.8:
    print("Accuracy is at least 0.8. Great job!")
else:
    print("Accuracy is below 0.8. Consider re-evaluating your model or features.")
