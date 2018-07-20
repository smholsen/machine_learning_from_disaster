from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
import feature_eng
import pandas as pd


# Scoring function (Accuracy)
def score(m_classifier, X, y):
    cross_value_score = cross_val_score(m_classifier, X, y, scoring='accuracy')
    return np.mean(cross_value_score)


# Get the expanded training and testing set from our feature engineering
train, test, targets = feature_eng.get_train_test_targets()
# Create the Random Forest Classifier
parameters = {'n_estimators': 100, 'max_features': 'auto', 'criterion': 'gini',
              'min_samples_split': 2, 'min_samples_leaf': 2}
classifier = RandomForestClassifier(**parameters)

# Fit it
classifier.fit(train, targets)


# print('Score:', score(RandomForestClassifier(), train, targets))
output = classifier.predict(test).astype(int)
df = pd.DataFrame()
tmp = pd.read_csv('data/test.csv')
df['PassengerId'] = tmp['PassengerId']
df['Survived'] = output
df[['PassengerId', 'Survived']].to_csv('data/results.csv', index=False)
