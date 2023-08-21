import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
   dt_heart = pd.read_csv('data/heart.csv')
 
   # Imprimimos un encabezado con los primeros 5 registros
#    print(dt_heart.head(5))
 
   # Guardamos nuestro dataset sin la columna de target
   dt_features = dt_heart.drop(['target'], axis=1)
   # Este será nuestro dataset, pero sin la columna
   dt_target = dt_heart['target']
 
   # Normalizamos los datos
   dt_features = StandardScaler().fit_transform(dt_features)
#    print(dt_features)
  
   # Partimos el conjunto de entrenamiento. Para añadir replicabilidad usamos el random state
   X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

   kpca = KernelPCA(n_components=4, kernel='poly')
   kpca.fit(X_train)

dt_train = kpca.transform(X_train)
dt_test = kpca.transform(X_test)
logistic = LogisticRegression(solver='lbfgs')

logistic.fit(dt_train,y_train)
print('SCORE KPCA: ', logistic.score(dt_test, y_test))


