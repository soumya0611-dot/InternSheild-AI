import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Load Dataset
df=pd.read_csv('data/completeSpamAssassin.csv')

##Data Cleaning
#Remove missing values
df.dropna(inplace=True)

#Remove duplicate records
df.drop_duplicates(inplace=True)

#check again
print(df.shape)
print(df.isnull().sum())

#Features and Target
X= df["Body"]
y=df["Label"]

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#Text to numerical form
tfidf_vectorizer = TfidfVectorizer( stop_words='english',max_features=50000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Model
model= RandomForestClassifier(
n_estimators =100,
random_state=42
)
#Training
model.fit(X_train_tfidf, y_train)

#Evaluation
y_pred=model.predict(X_test_tfidf)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test,y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test,y_pred))

#Save model
joblib.dump(model, "spam_model.joblib")
joblib.dump(tfidf_vectorizer,"tfidf_vectorizer.joblib")
print("Model Saved successfully")







