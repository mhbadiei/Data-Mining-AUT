import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns, matplotlib.pyplot as plt, operator as op


class KmeansWorms:
    def __init__(self ,file, cluster_num , max_itr):
        with open(file, newline='') as csvfile:
            data = np.asarray(list(csv.reader(csvfile)))
        self.data = data
        self.output = []
        self.cdata = self.data[np.array([x for x in range(1,self.data.shape[0])]),:]
        self.pure_data = self.cdata[:,np.array([x for x in range(1,self.data.shape[1])])]
        self.C =[]
        self.cluster_num = cluster_num
        self.max_itr = max_itr
        self.clusters = []
        self.x = []
        self.y = []

        plt.figure()
        plt.scatter(self.pure_data[:,0].astype(float), self.pure_data[:,1].astype(float), s=0.01)
        plt.title('worms')
        plt.axis('off')
        
    def setRand(self):
        for i in range(self.cluster_num):
            self.C.append(self.pure_data[np.random.randint(self.pure_data.shape[0])].astype(float))    
        #print(self.C)
        
    def dist(self,p1,p2):
        return np.sqrt(np.sum((p2-p1)**2))
        
    def centerOfMass(self , cluster_data_arr):
        
        cluster_data_arr = np.asarray(cluster_data_arr)
        #print(cluster_data_arr.shape)
        data_arr = cluster_data_arr[:,np.array([x for x in range(1,cluster_data_arr.shape[1])])]
        data_arr=data_arr.astype(float)
        s = 0;
        n = 0;
		# print(data_arr)
        for i in data_arr:
            s = s + np.float64(i)
            n = n+1;
		# print(s)
        if (n==0):
            return np.array([-1000,-1000,-1000])
        #print((1.0/n)*s)
        return (1.0/n)*s;
        
    def train(self):
        self.setRand()
        for i in range(self.max_itr):
            self.train_()
     
    def train_(self):
        self.lastClusters = self.clusters.copy
        self.clusters = [[] for i in range(self.cluster_num)];
        self.x = [[] for i in range(self.cluster_num)];
        self.y = [[] for i in range(self.cluster_num)];
        for p_data in self.cdata:
            #print(p_data.shape)
            pure_data = p_data[np.array([x for x in range(1,p_data.shape[0])])]
            pure_data = pure_data.astype(float)
            d , last_d = 0 , 100000;
            nearest_ci = 0;
            for ci in range(self.cluster_num):
                d = self.dist(pure_data, self.C[ci])
                if d < last_d:
                    last_d = d ;
                    nearest_ci = ci ;
            self.clusters[nearest_ci].append(p_data)
            #print(np.asarray(p_data[p_data.shape[0]-1]))
            self.x[nearest_ci].append(pure_data[0])
            self.y[nearest_ci].append(pure_data[1])
        self.clusters = np.array(self.clusters)
        for ci in range(self.cluster_num): 
            if len(self.clusters[ci]) > 0:
                self.C[ci] = self.centerOfMass(self.clusters[ci])
                
    def printClusters(self):
        color=cm.rainbow(np.linspace(0,1,self.cluster_num))
        plt.figure()    
        for i,c in zip(range(self.cluster_num),color):
            #print('cluster No.' + str(i+1) + ' : ') 
            #print(self.output_clusters[i])
            #print('\n')
            plt.scatter(self.x[i],self.y[i], s=0.01, color=c);
        plt.axis('off')    
        plt.title('Kmeans on worms')
        plt.show()
            
    def avgOfDistance(self):
        sum = 0;
         
        for ci in range(self.cluster_num):        
            cal_avg_data = np.asarray(self.clusters[ci])
            cal_avg_data_arr = cal_avg_data[:,np.array([x for x in range(1,cal_avg_data.shape[1])])]
            cal_avg_data_arr=cal_avg_data_arr.astype(float)
            
            for data in cal_avg_data_arr:
                sum = sum + abs(self.dist(data, self.C[ci]))
        return sum/self.pure_data.shape[0]


def main():
    model = KmeansWorms('datasets/worms.csv',4,2)
    model.train();
    model.printClusters();

main();
