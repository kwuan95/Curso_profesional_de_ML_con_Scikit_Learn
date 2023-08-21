import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
   dt = pd.read_csv('data/candy-data.csv')
#    print(dt.head())

   X = dt.drop('competitorname', axis=1)

   kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
   print('Total de centros: ', len(kmeans.cluster_centers_))
   print('='*64)
   print(kmeans.predict(X))

   dt['group'] = kmeans.predict(X)
   print(dt.head())

   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.pairplot(dt[['sugarpercent','pricepercent','winpercent','group']], hue = 'group')
   plt.show()

