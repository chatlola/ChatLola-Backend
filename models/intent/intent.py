import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

path = "models/intent/Intent Dataset.csv"
df = pd.read_csv(path, on_bad_lines='error', encoding='unicode_escape')

#check number of utterances per intent
print(df['intent'].value_counts())

#drop all NAN rows
df = df.dropna()
df = df.drop_duplicates()

#check number of utterances per intent
print(df['intent'].value_counts())

#removes punctuation
df["utterance"] = df["utterance"].str.translate(
    str.maketrans('', '', string.punctuation)
)

#lowercase
df['utterance'] = df['utterance'].str.lower()

#MODEL TRAINING

#split the dataset
X = df['utterance']
y = df['intent']

X_train, X_test, y_train, y_test= train_test_split(X,y, train_size=0.8,
                                                   random_state=None,
                                                   shuffle=True, stratify=y)

#feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
x_test_tfidf = tfidf_vectorizer.transform(X_test)

#train Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(x_train_tfidf, y_train)

#save model
save_path = "models/intent/"

joblib.dump(naive_bayes, save_path + "intent_model.pkl")
joblib.dump(tfidf_vectorizer, save_path + "intent_tfidf_vectorizer.pkl")

#MODEL EVALUATION

# predict the y of the testing_data
y_pred_naive = naive_bayes.predict(x_test_tfidf)

#accuracy
accuracy = accuracy_score(y_test, y_pred_naive)
print("\nAccuracy:", accuracy)

# print/generate classification report
report = classification_report(y_test, y_pred_naive)
print("\nClassification Report:")
print(report)

#print specific rows in the data that are misclassified
#for actual, pred, sample in zip(y_test, y_pred_naive, X_test):
#    if actual != pred:
#        print(f"Text: {sample} | Actual: {actual} | Predicted: {pred}")
        