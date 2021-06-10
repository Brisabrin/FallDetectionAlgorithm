import numpy as np
import pandas as pd
import sklearn
from sktime.datasets import load_airline
from sklearn.model_selection import train_test_split
import traces
import datetime 
from  frozendict import frozendict 

df = pd.read_csv('CompleteDataSet.csv')

# n, m = df.shape
# x = []
# y = []


# transformer = TSFreshFeatureExtractor(default_fc_parameters="minimal")

ind = df.columns.get_loc("BeltAccelerometer")
sInd = df.columns.get_loc("BeltAngularVelocity")

# titanic.loc[titanic["Age"] > 35, "Name"]


n =  -1 
nf = {} 

# subject , activity , trial : 18 , 12 ,4 
for z in range(1 , 2) : 
    for i in range(1,2) : 
        for j in range(1,2 ) :
            
            if z == 5 or z == 9 : 
                continue 
            f = df.iloc[: , [ 0 , 15 ,16 ,17, 18, 19,20 ]][(df['Activity'] == i )& (df['Trial'] == j)& (df['Subject'] == z )]
            
            row , col  = f.shape
            n = max(n , row)
            
            
            #convert to regular time intervals 
            arr =  [float ( "{0:.6f}".format(0))] 
            for k in range(1 ,row) : 
    
                if k == 0 : 
                    # dic[0] = {0}
                    continue 
                
                s1 = f.iloc[k , 0]
    
                s1 = float(s1[s1.rfind(":") + 1 : ])
                # print(s1) 
                s2 = f.iloc[k-1, 0 ]
                
                s2 = float(s2[s2.rfind(":")+ 1 : ])
                #  ,float(f.iloc[k , 2]),float(f.iloc[k,3]),float(f.iloc[k ,4]) ,float(f.iloc[k,5]),float(f.iloc[k ,6])
                arr.append( float("{0:.6f}".format( arr[ len(arr) - 1  ] +  ( s1 - s2 )))  )
            
            nf[(z,i,j)] = []
            nf[(z,i,j)].append(arr)
            # print(arr[-1])
            for k in range( 1,2 ) : 
                print("HELLOOOO")
                print()
                print() 
                print()
                # dic = traces.TimeSeries()
                a = [] 
                for g in range(row) : 
                    a.append((arr[g] , float(f.iloc[g ,k])))
        
                dic = traces.TimeSeries(data = a) 
                
                #0.01 / 0.001 
                reg = dic.moving_average(start = arr[0] ,  sampling_period = 0.01, placement='left' , pandas = True) 
                # reg = dic.moving_average(pandas= True , sampling_period = 0.00000001)
        
                
                # print(len( reg ) )
                
                a = [ ] 
                # for g in range(len(reg) )  : 
                # nf[(z,i,j)].append(list(reg))  
                
                
            # dic[s1-s2] = float(f.iloc[k,1])     
