from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test_with_no_labels.csv')

def carbonCopy(df):
    
    """
    This function creates a copy of the original train data and 
    renames the classes, converting them from numbers to words
    
    Input: 
    df: original dataframe
        datatype: dataframe
    
    Output:
    df: modified dataframe
        datatype: dataframe 
        
    """
    sentiment = df['sentiment']
    word_sentiment = []

    for i in sentiment :
        if i == 1 :
            word_sentiment.append('Pro')
        elif i == 0 :
            word_sentiment.append('Neutral')
        elif i == -1 :
            word_sentiment.append('Anti')
        else :
            word_sentiment.append('News')

    df['sentiment'] = word_sentiment
        
    return df

train_df = carbonCopy(df)

X = train_df['message']
y = train_df['sentiment']


# Split the train data to create validation dataset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# Random Forest Classifier
rf = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', RandomForestClassifier(max_depth=5, 
                                              n_estimators=150))])

# K-NN Classifier
knn = Pipeline([('tfidf', TfidfVectorizer()),
                ('clf', KNeighborsClassifier(n_neighbors=5, 
                                             metric='minkowski', 
                                             p=2))])

# Logistic Regression
lr = Pipeline([('tfidf',TfidfVectorizer()),
               ('clf',LogisticRegression(C=1, 
                                         class_weight='balanced', 
                                         max_iter=1000))])





# Lets Retrain linear SVC using optimal hyperparameters:
lsvc_op = Pipeline([('tfidf', TfidfVectorizer(max_df=0.8,
                                                    min_df=2,
                                                    ngram_range=(1,2))),
                  ('clf', LinearSVC(C=.5, max_iter=3000))])


# Random forest 
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_valid)


# K - nearest neighbors
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_valid)

# Logistic Regression
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_valid)






# Define the number of folds for cross-validation
n_folds = 45

# Initialize StratifiedKFold with the number of folds
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=98)

# Perform cross-validation
fold_accuracies = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the pipeline on the training data
    lsvc_op.fit(X_train, y_train)
    
    # Predict labels on the test data
    y_pred = lsvc_op.predict(X_test)

# Refit and predict
lsvc_op.fit(X_train, y_train)
y_pred_1 = lsvc_op.predict(X_valid)



# save the model to disk
import pickle

rf_model = './finalized_model_rf.pkl'
pickle.dump(rf, open(rf_model, 'wb'))

kn_model = './finalized_model_kn.pkl'
pickle.dump(knn, open(kn_model, 'wb'))

lr_model = './finalized_model_lr.pkl'
pickle.dump(lr, open(lr_model, 'wb'))

LSVC_op = './finalized_model_lsvc_op.pkl'
pickle.dump(lsvc_op, open(LSVC_op, 'wb'))



