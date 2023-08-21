# metodos de ensamble
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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

   knn_class = KNeighborsClassifier().fit(X_train, y_train)
   knn_pred = knn_class.predict(X_test)
   print('='*64)
   print(accuracy_score(knn_pred, y_test))

   bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=100).fit(X_train, y_train)
   bag_pred = bag_class.predict(X_test)
   print('='*64)
   print(accuracy_score(bag_pred, y_test))





