import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns, matplotlib.pyplot as plt, operator as op

class Data:
    def __init__(self,file):
        self.player = pd.read_csv(file); 
       
    def printFirstLastRow(self):
        print ('First and last rows\n')
        flplayer = self.player.iloc[[0, -1]]
        print (flplayer,'\n')

    def printMissingValue(self):
        missingvalues = self.player.isnull() 
        print ('missing values\n')
        print (missingvalues)
        print ('number of missing values : ',missingvalues.sum().sum(),'\n')

    def printWeightCharacteristics(self):
        print('\n',self.player[self.player.columns[5]].describe(),'\n')
        #print(self.player.iloc[0,[0,1]])
        
    def printNationalityReport(self):
        print('\n') 
        nationality = {};
        for n in self.player.iloc[:,7]:
            if n not in nationality:
                nationality[n] = 1;
            else:
                nationality[n] = int(nationality[n]) + 1;
        sorted_nationality = {k: v for k, v in sorted(nationality.items(), key=lambda item: item[1])}
        
        print('nationality of manimum players')
        print(list(sorted_nationality)[1], sorted_nationality[list(sorted_nationality)[1]])
        print('nationality of maximum players')
        print(list(sorted_nationality)[-1], sorted_nationality[list(sorted_nationality)[-1]],'\n')
        
    def promisingPlayer(self):
        x = list(self.player.iloc[:,9])
        y = list(self.player.iloc[:,10])
        #print(len(x))
        for n in range(len(self.player)):      
            if int(x[n])>84 and int(y[n])>4:  
                print('\n', n, self.player.iloc[n,1],x[n],y[n])
        print('\n')    
    
    def plotPromisingPlayer(self):
        promising = {};
        x = list(self.player.iloc[:,9])
        y = list(self.player.iloc[:,10])
        for n in range(len(self.player)):      
            if int(x[n])>84 and int(y[n])>4:
                for p in self.player.iloc[n,14].split(','):
                    if p not in promising: 
                        promising[p] = [self.player.iloc[n,9]]
                    else:
                        promising[p].append(self.player.iloc[n,9])
        
        #print(promising)
        sorted_keys, sorted_vals = zip(*sorted(promising.items(), key=op.itemgetter(1)))

        # almost verbatim from question
        sns.set(context='notebook', style='whitegrid')
        sns.utils.axlabel(xlabel="BestPosition", ylabel="Potential", fontsize=16)
        sns.boxplot(data=sorted_vals, width=.48)
        #sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
        plt.xticks(plt.xticks()[0], sorted_keys)
        plt.show()
   
    def teamOfMaxPromisingPlayer(self):                
        promising = {};
        x = list(self.player.iloc[:,9])
        y = list(self.player.iloc[:,10])
        for n in range(len(self.player)):      
            if int(x[n])>84 and int(y[n])>4:
                p = self.player.iloc[n,15]
                if p not in promising: 
                    promising[p] = 1
                else:
                    promising[p] = int(promising[p]) + 1
        
        sorted_promising = {k: v for k, v in sorted(promising.items(), key=lambda item: item[1])}
        print(sorted_promising,'\n')
       
    def valueOfChelseaPromissingPlayers(self):
        value = 0;
        x = list(self.player.iloc[:,9])
        y = list(self.player.iloc[:,10])
        for n in range(len(self.player)):      
            if int(x[n])>84 and int(y[n])>4:
                p = self.player.iloc[n,15]
                if p == 'Chelsea': 
                    value = value + int(self.player.iloc[n,16])
                    #print(p)
        print(value)
        
    def contractUntilAndNotInNationalTeam(self):
        num = 0;
        i=0;
        x = list(self.player.iloc[:,20])
        y = list(self.player.iloc[:,24])
        #print(self.player.iloc[:,20])
        #print(self.player.iloc[:,24])
        for n in range(len(self.player)): 
            #i+=1
            if x[n]== 2021.0 and y[n]=='Not in team':
                #print(i)
                num +=1
                #print(self.player.iloc[n,1])
        print('\n', num, '\n')
   
    def taremiReport(self):
        report = self.player.loc[self.player['Name'] == 'M. Taremi'] 
        print('\n Mehdi Taremi reports\n')
        print('Positions')
        print(report.iloc[:,13], '\n')
        print('Best Position')
        print(report.iloc[:,14], '\n')
        print('WageEUR')
        print(report.iloc[:,17], '\n')
        print('Club')
        print(report.iloc[:,15])
        
        
if __name__ == "__main__":
    model=Data('players.csv');
    model.printFirstLastRow();
    model.printMissingValue();
    model.printWeightCharacteristics();
    model.printNationalityReport();
    model.promisingPlayer();
    model.plotPromisingPlayer();
    model.teamOfMaxPromisingPlayer();
    model.valueOfChelseaPromissingPlayers();
    model.contractUntilAndNotInNationalTeam();
    model.taremiReport();
    