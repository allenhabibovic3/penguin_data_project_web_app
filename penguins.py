import pandas as pd

df = pd.read_csv("penguin_data.csv")

categorical = ['sex', 'island']

for col in categorical:
    dummy_df = pd.get_dummies(df[col])
    df = pd.concat([df, dummy_df], axis=1)

    df.drop(columns=[col], inplace=True, axis=1)

df['species'] = df['species'].map({'Adelie':0, 'Chinstrap':1, 'Gentoo':2})

X = df.drop(columns=['species'], axis=1)
y = df['species']

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X, y)

import pickle
pickle.dump(clf, open('penguin_clf.pkl', 'wb'))