import matplotlib.pyplot as plt
from Kmean import *

if __name__ == '__main__':
    
    max_iteration = 10;
    x_axis_name = []
    y_axis_value = []
    
    for k in range(1,6):
        model = Kmean('datasets/iris.csv',k,max_iteration)
        model.train();
        
        #model.printClusters();
        #print("MAE Error : "+str(model.avgOfDistance())+"\n")
        
        x_axis_name.append(str(k))
        y_axis_value.append(model.avgOfDistance())
    
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(x_axis_name ,y_axis_value , color ='maroon',
            width = 0.4)
     
    plt.xlabel("No. of clusters")
    plt.ylabel("Mean absolute error (MAE)")
    plt.title("MAE corresponding to number of clusters")
    plt.show()    