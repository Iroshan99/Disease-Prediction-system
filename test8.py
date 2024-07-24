import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create the dataset
data = {
    'Disease': ['Flu', 'Common Cold', 'Diabetes', 'Hypertension'],
    'Symptom1': ['Fever', 'Runny Nose', 'Frequent Urination', 'Headache'],
    'Symptom2': ['Cough', 'Sneezing', 'Increased Thirst', 'Dizziness'],
    'Symptom3': ['Fatigue', 'Sore Throat', 'Blurred Vision', 'Nausea']
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('disease_symptoms.csv', index=False)

# Load the dataset
df = pd.read_csv('disease_symptoms.csv')

# Combine symptoms into a single column
df['Symptoms'] = df[['Symptom1', 'Symptom2', 'Symptom3']].values.tolist()

# One-hot encode the symptoms
mlb = MultiLabelBinarizer()
symptoms_encoded = mlb.fit_transform(df['Symptoms'])

# Prepare the final dataset for training
X = symptoms_encoded
y = df['Disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


def predict_disease(user_symptoms):
    user_symptoms_encoded = mlb.transform([user_symptoms])
    prediction = clf.predict(user_symptoms_encoded)
    return prediction[0]


if __name__ == "__main__":
    user_symptoms = input("Enter your symptoms separated by commas: ").split(',')
    user_symptoms = [symptom.strip() for symptom in user_symptoms]

    disease_prediction = predict_disease(user_symptoms)
    print(f"Based on your symptoms, you might have: {disease_prediction}")
