
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Load the customer data
df = pd.read_csv('Churn_Modelling.csv')

# Preprocess the data
label_encoder = LabelEncoder()
# Use one-hot encoding for categorical features
df = pd.get_dummies(df, columns=['Gender', 'Surname', 'Geography', 'Age', 'Tenure',
                                  'CreditScore', 'Balance', 'NumOfProducts',
                                  'HasCrCard', 'IsActiveMember'])

# Convert Exited to numeric
df['Exited'] = df['Exited'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited', axis=1), df['Exited'], test_size=0.25, random_state=0)

# Train the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} accuracy: {accuracy:.2f}')