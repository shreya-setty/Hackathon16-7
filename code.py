import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = 'dataset.csv'  
df = pd.read_csv(file_path)

df = df.dropna(subset=['Crm Cd'])

numerical_features = ['TIME OCC', 'Rpt Dist No', 'Vict Age', 'LAT', 'LON']
categorical_features = ['AREA NAME', 'Crm Cd Desc', 'Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc', 'Status', 'Status Desc']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

classifier = SVC()  
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

X = df[numerical_features + categorical_features]
y = df['Crm Cd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Mapping crime codes to descriptions
crime_code_to_desc = df[['Crm Cd', 'Crm Cd Desc']].drop_duplicates().set_index('Crm Cd')['Crm Cd Desc'].to_dict()

# Testing the model with a new case
test_case = {
    'Date Rptd': ['02-10-2021 00:00'],
    'DATE OCC': ['02-09-2021 00:00'],
    'TIME OCC': [1900],
    'AREA NAME': ['central'],
    'Rpt Dist No': [200],
    'Part 1-2': [1],
    'Crm Cd': [440],
    'Crm Cd Desc': ['robbery'],
    'Vict Age': [30],
    'Vict Sex': ['M'],
    'Vict Descent': ['H'],
    'Premis Desc': ['street'],
    'Weapon Used Cd': [300],
    'Weapon Desc': ['handgun'],
    'Status': ['AA'],
    'Status Desc': ['adult arrested'],
    'Crm Cd 1': [440],
    'LOCATION': ['1100 S Flower St'],
    'Cross Street': ['Broadway'],
    'LAT': [34.0455],
    'LON': [-118.2615]
}
test_df = pd.DataFrame(test_case)
X_test_case = test_df[numerical_features + categorical_features]

# Predicting the crime code
predicted_crm_cd = model.predict(X_test_case)[0]

# Mapping the predicted crime code to the description
predicted_crm_cd_desc = crime_code_to_desc.get(predicted_crm_cd, 'Unknown crime code')

print(f'Predicted Crime Description: {predicted_crm_cd_desc}')
