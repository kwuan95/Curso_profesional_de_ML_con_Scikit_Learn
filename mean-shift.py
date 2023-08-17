import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
   dt = pd.read_csv('data/candy-data.csv')
#    print(dt.head())

X = dt.drop('competitorname', axis=1)

meanshift = MeanShift().fit(X)
print(max(meanshift.labels_))
print('='*64)
print(meanshift.cluster_centers_)

dt['meanshift']= meanshift.labels_
print(dt.head())
