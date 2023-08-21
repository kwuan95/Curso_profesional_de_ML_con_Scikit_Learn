import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
   dt = pd.read_csv('data/2017.csv')
   #print(dt.describe())

   nombres_columnas = dt.columns.tolist()
   print(nombres_columnas)


   X = dt[['Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.', 'Freedom' ,'Generosity', 'Trust..Government.Corruption.', 'Dystopia.Residual']] 
   y = dt[['Happiness.Score']]  

   print(X.shape)
   print(y.shape)

   # Partimos el conjunto de entrenamiento. Para añadir replicabilidad usamos el random state
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

   modelLiniar = LinearRegression().fit(X_train,y_train)
   y_predict_linear = modelLiniar.predict(X_test)

   modelLasso = Lasso(alpha=0.02).fit(X_train,y_train)
   y_predict_lasso = modelLasso.predict(X_test)

   modelRidge = Ridge(alpha=1).fit(X_train,y_train)
   y_predict_ridge = modelRidge.predict(X_test)

   modelElasticNet = ElasticNet(alpha=0.02, l1_ratio=0.5).fit(X_train,y_train)
   y_predict_elasticnet = modelElasticNet.predict(X_test)

   linear_loss = mean_squared_error(y_test,y_predict_linear)
   print("Linear Loss: ", linear_loss) 

   lasso_loss = mean_squared_error(y_test,y_predict_lasso)
   print("Lasso Loss: ", lasso_loss)

   ridge_loss = mean_squared_error(y_test,y_predict_ridge)
   print("Ridge Loss: ", ridge_loss)

   elasticnet_loss = mean_squared_error(y_test,y_predict_elasticnet)
   print("ElasticNet Loss: ", elasticnet_loss)

   print('='*32)
   print('Coef Lasso')
   print(modelLasso.coef_)


   print('='*32)
   print('Coef Ridge')
   print(modelRidge.coef_)

#    print('='*32)
#    print('Coef elasticnet')
#    print(elasticnet_loss.coef_)

import pandas as pd

# Crear un dataframe con los coeficientes de Lasso
df_lasso = pd.DataFrame({'columna': X.columns, 'coeficiente': modelLasso.coef_})

# Crear un dataframe con los coeficientes de Ridge
df_ridge = pd.DataFrame({'columna': X.columns, 'coeficiente': modelRidge.coef_[0]})

# Crear un dataframe con los coeficientes de ElasticNet
df_elasticnet = pd.DataFrame({'columna': X.columns, 'coeficiente': modelElasticNet.coef_})

# Imprimir los dataframes
print('Coeficientes Lasso:')
print(df_lasso)   
print('='*32)


print('Coeficientes Ridge:')
print(df_ridge)
print('='*32)


print('Coeficientes ElasticNet:')
print(df_elasticnet)

