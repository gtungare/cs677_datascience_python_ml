# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 1 to Question 5
Description of Problem (just a 1-2 line summary!)
    
######### READ ME : Flow of the script #####################    
    
Please note all question are given in one script 

"""

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls'
data = pd.read_excel(url, sheet_name='Raw Data',index_col=0,keep_default_na=False )
df1 = pd.DataFrame(data)
df=df1[1:2127]

np.random.seed(10001)


try:   
    
            
    ## Quesion 1
 
    print("Question  1 part 1")
    print("\n")
    print(df.head(5))
    print("\n")
    
    df_grp = df[['ASTV','MSTV','Max','Median','NSP']]
    
    # combine NSP labels into two groups: N (normal - these
    #labels are assigned) and Abnormal (everything else) 
    #We will use existing class 1 for normal and dene class 0 forabnormal.
    
    # Defining the labels as needed
    
    df_grp['CLASS'] = np.where(df_grp["NSP"] == 1,1,0)
    df_grp2 = df_grp[['ASTV','MSTV','Max','Median','CLASS']]
    
    print("Question  1 part 2")
    print(df_grp2.head(5)) 
    
    # creating a master dataframe for final summary
    
    df_log_mstr = pd.DataFrame({'Model' : [],
                        'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })    
   
    ######################## naive_bayes
    
    df_log_reg = pd.DataFrame({'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })    
    
        
    X = df_grp2[['ASTV','MSTV','Max','Median']].values
    Y = df_grp2[['CLASS']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
        
    NB_classifier = GaussianNB().fit(X_train, Y_train.ravel())
    prediction = NB_classifier.predict(X_test)     
    
    print("\n") 
    
    print(" Naive Bayes - Summary ")    
    print("\n")
    print("Question 2 - part 2 - accuracy")    
    accuracy=accuracy_score(Y_test,prediction)
    print("\n")    
    print("accuracy naive bayes :",round(accuracy,6))    
    print("\n")  
    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print("Question 2 - part 3 - Summary")    
    print("\n")
    print(df_log_reg)    
    
    ## collecing in master dataframe for final summary
    df_log_mstr.at[0,'Model']='Naive Bayes'
    df_log_mstr.at[0,'TP']=tp
    df_log_mstr.at[0,'FP']=fp
    df_log_mstr.at[0,'TN']=tn
    df_log_mstr.at[0,'FN']=fn
    df_log_mstr.at[0,'Accuracy']=accuracy 
    df_log_mstr.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    ######################## Decision Tree
    print("\n")    
    print(" Decision Tree - Summary ")   
    print("\n")
    
    X = df_grp2[['ASTV','MSTV','Max','Median']].values
    Y = df_grp2[['CLASS']].values
    print("Question 3 - part 1 - splitting the data set")   
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
        
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf = clf.fit(X_train,Y_train)
    prediction = clf.predict(X_test)
    
    accuracy=accuracy_score(Y_test,prediction)
    
    print("Question 3 - part 2 - accuracy")    
    print("\n")
    
    print("accuracy  Decision Tree :",round(accuracy,6))    
    print("\n")    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print("Question 3 - part 3 - Summary")    
    print("\n")
    print(df_log_reg)
    print("\n")    
    
    ## collecing in master dataframe for final summary
    df_log_mstr.at[1,'Model']='Decision Tree'
    df_log_mstr.at[1,'TP']=tp
    df_log_mstr.at[1,'FP']=fp
    df_log_mstr.at[1,'TN']=tn
    df_log_mstr.at[1,'FN']=fn
    df_log_mstr.at[1,'Accuracy']=accuracy 
    df_log_mstr.at[1,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[1,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    ######################## Random Forest 
        

   ### Putting the result to dataframe for easy reading    
    dfn = pd.DataFrame({'N' : [],
                        'd' : [],
                        'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'error_rate' : [],
                        'TPR' : [],
                        'TNR' : []                        
                        })
    
    
    # Defining variable for the counter and best reutlt 
    temp_accur=0
    best_N=0
    best_d=0
    best_c=0
    ## defefining a counter
    c=0
    # for N from 1 to 10
    my_list=range(1,11)
    # using tuple unpacking
    for i,k in enumerate(my_list):
        # for d from 1 to 5
        for d in range (1,6):
            c=c+1
            
            X = df_grp2[['ASTV','MSTV','Max','Median']].values
            Y = df_grp2[['CLASS']].values
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    
            model = RandomForestClassifier(n_estimators =k, max_depth =d,
                               criterion ='entropy')

            model.fit(X_train,Y_train.ravel())
            prediction = model.predict(X_test)
            error_rate = np.mean(prediction != Y_test)
            accuracy=accuracy_score(Y_test,prediction) 
            ## keeping count of best accuracy N K 
            if accuracy>temp_accur:
                temp_accur=round(accuracy,6)
                best_N=k
                best_d=d
                best_c=c
                temp_accur=round(accuracy,6)
            else:
                pass
            
            
            #print(best_acc,best_N,best_d)
            tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
            dfn.at[c,'N']=k
            dfn.at[c,'d']=d
            dfn.at[c,'TP']=tp
            dfn.at[c,'FP']=fp
            dfn.at[c,'TN']=tn
            dfn.at[c,'FN']=fn
            dfn.at[c,'Accuracy']=accuracy 
            dfn.at[c,'error_rate']=error_rate 
            dfn.at[c,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
            dfn.at[c,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    
    ## collecing in master dataframe for final summary for best result
    
    df_log_mstr.at[2,'Model']='Random Forest'
    df_log_mstr.at[2,'TP']=dfn.loc[best_c]['TP']
    df_log_mstr.at[2,'FP']=dfn.loc[best_c]['FP']
    df_log_mstr.at[2,'TN']=dfn.loc[best_c]['TN']
    df_log_mstr.at[2,'FN']=dfn.loc[best_c]['FN']
    df_log_mstr.at[2,'Accuracy']=dfn.loc[best_c]['Accuracy']
    df_log_mstr.at[2,'TPR']=dfn.loc[best_c]['TPR']
    df_log_mstr.at[2,'TNR']=dfn.loc[best_c]['TNR']
    
    print("Question 4 - part 2 - Printing the plot")    
    dfn.plot(y='error_rate',x='N',xlabel='N',ylabel='error_rate')
    print("\n")    
    print("Random Forest - Summary ")    
    print("\n")    
    print("Question 4 - part 1 - Summary")    
    print("\n")    
    print(dfn)
    print("\n")   
    print("Question 4 - part 3 - what is the accuracy for the best combination of N and k?")    
    print("Comments - THe best accuracy is 91.4393% for combination of N=8 and d=4 ")
    print("Best accuracy-",temp_accur,"Best N-",best_N,"Best d",best_d)
    print("\n")    
    print("Question 4 - part 4 - compute the confusion matrix using the best combinationof N and d")    
    print("Comments - given in summary above")
    print(df_log_mstr[df_log_mstr.index==2])
    
    ######################## 
    print("\n")    
    print("Question 5 - final summary")    
    
    print(df_log_mstr)
    print("\n") 
    print("Observation - 1 - Random Forest is performing best among three classifiers")    
    print("Observation - 2 - TPR,Accuracy for Random Forest is the highest among three classifiers" )    
    print("Observation - 3 - Overall across all parameters Random Forest performing better for this paticular  dataset" )    
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ')    