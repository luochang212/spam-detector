import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


nltk.data.path.append("./data/nltk_data")


class SpamDetector:

    def __init__(self):
        pass

    def read_data(self, file_path):
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        ps = PorterStemmer()
        df = df[['v1', 'v2']]
        df['v1'] = df['v1'].map({'spam': 1, 'ham': 0})
        df['v2'] = df['v2'].map(lambda sentence: ' '.join([ee for ee in [ps.stem(e) for e in word_tokenize(sentence)] if ee not in stopwords.words('english')]))

        return df

    def train(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df.v2, df.v1, test_size=0.33, random_state=42)

        cv = CountVectorizer()
        X_train = cv.fit_transform(X_train).astype(int)
        y_train = y_train.astype(int)

        clf = MultinomialNB(force_alpha=True)
        clf.fit(X_train, y_train)
        
        clf_pred= clf.predict(cv.transform(X_test))
        print(clf_pred)
        score = f1_score(y_test, clf_pred)
        print(score)


    def main(self):
        df = self.read_data(file_path='./data/spam.csv')
        self.train(df)


if __name__ == '__main__':
    s = SpamDetector()
    s.main()
