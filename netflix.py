import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files

uploaded = files.upload()  # Upload your CSV here

for filename in uploaded.keys():
    df = pd.read_csv(filename)

df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df['Year'] = df['Release_Date'].dt.year

df['Director'].fillna('Unknown', inplace=True)
df['Cast'].fillna('Unknown', inplace=True)
df['Country'].fillna('Unknown', inplace=True)
df['Rating'].fillna('NR', inplace=True)

le_category = LabelEncoder()
df['Category_encoded'] = le_category.fit_transform(df['Category'])

le_type = LabelEncoder()
df['Type_encoded'] = le_type.fit_transform(df['Type'])

le_country = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['Country'])

def extract_duration(x):
    if pd.isna(x):
        return 0
    if 'min' in str(x):
        return int(x.replace(' min',''))
    elif 'Season' in str(x):
        return int(x.split(' ')[0])
    else:
        return 0

df['Duration_num'] = df['Duration'].apply(extract_duration)

plt.figure(figsize=(12,6))
sns.countplot(x='Year', hue='Category', data=df)
plt.title('Movies vs TV Shows Over Years')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(y='Type', order=df['Type'].value_counts().index, data=df)
plt.title('Most Common Genres')
plt.show()

top_countries = df['Country'].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Contributing Countries')
plt.show()

features = ['Type_encoded', 'Country_encoded', 'Duration_num']
X = df[features]
y = df['Category_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_category.classes_))

feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances)