# metodos de ensamble
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
   dt_heart = pd.read_csv('data/heart.csv')
#    print(dt_heart['target'].describe())

   # Guardamos nuestro dataset sin la columna de target
   X = dt_heart.drop(['target'], axis=1)
   # Este será nuestro dataset, pero sin la columna
   y = dt_heart['target']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

   boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
   boost_pred = boost.predict(X_test)
   print('='*64)
   print(accuracy_score(boost_pred, y_test))