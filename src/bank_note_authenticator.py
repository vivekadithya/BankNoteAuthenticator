import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


# Load Data
df = pd.read_csv('BankNote_Authentication.csv')

# Split Dependent and Independent Variables
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Generate Training and Testing Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Implement Random Forest Classfier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

# Prediction
y_pred = classifier.predict(x_test)

# Validate Accuracy
score = accuracy_score(y_test, y_pred)
print(f"The trained model's accuracy is {round(score*100,2)}%")

# Serialize the model for deployment
pickle_out = open('bank_note_classifier.pkl', 'wb')
pickle.dump(classifier, pickle_out)
pickle_out.close()
