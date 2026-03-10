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

path = "model/Intent Dataset.csv"
intent_df = pd.read_csv(path, on_bad_lines='error')

#check number of utterances per intent
print(intent_df['intent'].value_counts())

#drop all NAN rows
intent_df = intent_df.dropna()

#removes punctuation
intent_df["utterance"] = intent_df["utterance"].str.translate(
    str.maketrans('', '', string.punctuation)
)

#lowercase
intent_df['utterance'] = intent_df['utterance'].str.lower()

#MODEL TRAINING

#split the dataset
intent_X = intent_df['utterance']
intent_y = intent_df['intent']
intent_X_train, intent_X_test, intent_y_train, intent_y_test= train_test_split(intent_X,intent_y, train_size=0.8,
                                                   random_state=None,
                                                   shuffle=True, stratify=intent_y)

#feature extraction using TF-IDF
intent_tfidf_vectorizer = TfidfVectorizer()
intent_x_train_tfidf = intent_tfidf_vectorizer.fit_transform(intent_X_train)
intent_x_test_tfidf = intent_tfidf_vectorizer.transform(intent_X_test)

#train Naive Bayes classifier
intent_naive_bayes = MultinomialNB()
intent_naive_bayes.fit(intent_x_train_tfidf, intent_y_train)

#save model
save_path = "chatlola/"

joblib.dump(intent_naive_bayes, save_path + "intent_model.pkl")
joblib.dump(intent_tfidf_vectorizer, save_path + "intent_tfidf_vectorizer.pkl")

#MODEL EVALUATION

# predict the y of the testing_data
intent_y_pred_naive = intent_naive_bayes.predict(intent_x_test_tfidf)

#accuracy
intent_accuracy = accuracy_score(intent_y_test, intent_y_pred_naive)
print("\nAccuracy:", intent_accuracy)

# print/generate classification report
intent_report = classification_report(intent_y_test, intent_y_pred_naive)
print("Classification Report:")
print(intent_report)