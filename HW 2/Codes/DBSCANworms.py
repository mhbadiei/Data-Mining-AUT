import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns, matplotlib.pyplot as plt, operator as op
from sklearn.cluster import DBSCAN

class DBSCANworms:
    def __init__(self ,file, cluster_num , max_itr):
        with open(file, newline='') as csvfile:
            data = np.asarray(list(csv.reader(csvfile)))
        self.data = data
        self.cdata = self.data[np.array([x for x in range(1,self.data.shape[0])]),:]
        self.pure_data = self.cdata[:,np.array([x for x in range(1,self.data.shape[1])])]
        f = list(zip(self.pure_data[:,0].astype(float), self.pure_data[:,1].astype(float)));
        #print(f)
        
        clustering = DBSCAN(eps=20, min_samples=36).fit(f)
        y_pred = clustering.fit_predict(f)
        f = np.array(f)
        plt.scatter(f[:,0], f[:,1],c=y_pred, cmap='Paired',s=0.002)
        plt.title("DBSCAN")
        #print(clustering.labels_)

        plt.show()


def main():
    model = DBSCANworms('datasets/worms.csv',4,2)


main();
