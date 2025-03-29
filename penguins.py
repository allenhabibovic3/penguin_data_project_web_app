import pandas as pd


import subprocess
import sys

def install_distutils():
  """Attempts to install the 'distutils' package (though it's deprecated)."""
  try:
    # Attempt to install, but distutils is now part of setuptools, or removed.
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'setuptools'])
    print("setuptools installed or already present. distutils functionality included.")

  except subprocess.CalledProcessError as e:
    print(f"Error installing setuptools: {e}")
    print("distutils is deprecated and may not be installable directly.")
    print("It is typically included in setuptools, or is part of the standard library in older python versions.")
    print("Consider updating your dependencies to avoid reliance on distutils.")

install_distutils()

import setuptools
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
