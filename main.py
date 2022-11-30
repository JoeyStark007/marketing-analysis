import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skylearn

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# total observations = 2205
# We are doing 20% train and 80% test
# train observations = 441
# test observations = 1764

# import
my_performance_train  = pd.read_csv("C:/Users/jgtef/PycharmProjects/MarketingAnalytics/Channel_Performance_train.csv",
                      sep =",",  # delimiter
                     header= 0, # header in first row
                     index_col=0 # ids in first column
)

# Normalize the data
my_performance_train = (my_performance_train - my_performance_train.mean())/my_performance_train.std()
my_performance_train.head()

# Cluster with k-Means
mykmeans = KMeans(n_clusters=3).fit(my_performance_train)
mykmeans.labels_

# Evaluate clusters using PCA

mypca = PCA(n_components=2)
myscores = mypca.fit_transform(my_performance_train)

myscores = pd.DataFrame(myscores, columns = ["PC1", "PC2"])
sns.scatterplot(data=myscores, x = "PC1", y = "PC2", c = mykmeans.labels_)

myloadings = pd.DataFrame(mypca.components_.T, index= my_performance_train.columns, columns= ["PC1", "PC2"])

print(myloadings)






