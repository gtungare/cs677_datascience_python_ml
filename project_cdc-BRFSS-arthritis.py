# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  Project 
Description of Problem (just a 1-2 line summary!): Create summary table for each ticker
    
######### READ ME : Flow of the script #####################    
    
    
     >>> Please expand the right Console the see the output properly  
     
     >>> 2018 BRFSS Survey Data prepared by CDC 


"""

import os
import csv
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm

projfilename='project-2018-BRFSS-arthritis'
here =os.path.abspath( __file__ )
input_dir =os.path.abspath(os.path.join(here ,os. pardir ))
proj_file = os.path.join(input_dir, projfilename + '.csv')

np.random.seed(10001)


try:   
    
            
    ## Read CDC 2018-BRFSS-arthritis data into dataframe

    data = pd.read_csv(proj_file,delimiter=',')
    df = pd.DataFrame(data)

    ## Printing the datafame for reference 
    
    print(len(df))    
    print("Printing the datafame for reference") 
    print("\n")
    print(df.head(5)) 
    
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
    
    ### Putting the result to dataframe for easy reading    
    dfn = pd.DataFrame({'K' : [],
                        'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })
    accuracy = []
    for k in range (1,28,2):
        
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        
        X = df[list(df.columns.values)[1:108]].values
        Y = df[['havarth3']].values
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
        knn_classifier.fit( X_train,Y_train.ravel())
        yprediction = knn_classifier.predict(X_test)    
        accuracy=accuracy_score(Y_test,yprediction)
        tn, fp, fn, tp = confusion_matrix(Y_test, yprediction).ravel()
        
        dfn.at[k,'K']=k
        dfn.at[k,'TP']=tp
        dfn.at[k,'FP']=fp
        dfn.at[k,'TN']=tn
        dfn.at[k,'FN']=fn
        dfn.at[k,'Accuracy']=accuracy 
        dfn.at[k,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
        dfn.at[k,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    print("\n")    
    
    print(" KNN - Summary ")    
    print(dfn)


    #__________________________ Logistic Regression _________________________

    
    df_log_reg = pd.DataFrame({'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })
    log_reg_classifier  = LogisticRegression()
    X = df[list(df.columns.values)[1:108]].values
    Y = df[['havarth3']].values
    #scaler = StandardScaler()
    #scaler.fit(X)
    #X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
    log_reg_classifier.fit( X_train,Y_train.ravel())
    yprediction = log_reg_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)
      
    print("\n")    
    
    print(" Logistic Regression- Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, yprediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)
    
        
    ## collecing in master dataframe for final summary
    df_log_mstr.at[1,'Model']='Logistic Regression'
    df_log_mstr.at[1,'TP']=tp
    df_log_mstr.at[1,'FP']=fp
    df_log_mstr.at[1,'TN']=tn
    df_log_mstr.at[1,'FN']=fn
    df_log_mstr.at[1,'Accuracy']=accuracy 
    df_log_mstr.at[1,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[1,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    
    # _____________________________ naive_bayes _____________________________
    
    
    from sklearn.naive_bayes import GaussianNB
        
    X = df[list(df.columns.values)[1:108]].values
    Y = df[['havarth3']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
        
    NB_classifier = GaussianNB().fit(X_train, Y_train.ravel())
    prediction = NB_classifier.predict(X_test)     
    
    accuracy=accuracy_score(Y_test,prediction)
    
    print("\n")    
    
    print(" Naive Bayes - Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)    
    
    
    
    ## collecing in master dataframe for final summary
    df_log_mstr.at[2,'Model']='Naive Bayes'
    df_log_mstr.at[2,'TP']=tp
    df_log_mstr.at[2,'FP']=fp
    df_log_mstr.at[2,'TN']=tn
    df_log_mstr.at[2,'FN']=fn
    df_log_mstr.at[2,'Accuracy']=accuracy 
    df_log_mstr.at[2,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[2,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
   
    
    #_________________________ Random Forest __________________________
        
    from sklearn.ensemble import RandomForestClassifier
    

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
    
    
    # Defining variable for the counter 
    temp_accur=0
    best_N=0
    best_d=0
    best_c=0
    ## defefining a counter
    c=0
    # for N from 1 to 10
    my_list=range(1,11)
    
    X = df[list(df.columns.values)[1:108]].values
    Y = df[['havarth3']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
    
    # using tuple unpacking
    for i,k in enumerate(my_list):
        # for d from 1 to 5
        for d in range (1,6):
            c=c+1
            
            
    
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
    
    print("Random Forest - Summary - with differnet value of N and estimation paramenter d")    
    print("\n")
    print(dfn)  
    
    #____________________ linear kernel SVM ___________________________
    
    X = df[list(df.columns.values)[1:108]].values
    Y = df[['havarth3']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)
    
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(X_train, Y_train.ravel())
    
    prediction = svm_classifier.predict(X_test)

    accuracy=accuracy_score(Y_test,prediction)
    
    print("\n")    
    
    print(" linear kernel SVM - Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)
    print("\n") 
    
       
   ## collecing in master dataframe for final summary
    df_log_mstr.at[3,'Model']='Linear Kernel SVM'
    df_log_mstr.at[3,'TP']=tp
    df_log_mstr.at[3,'FP']=fp
    df_log_mstr.at[3,'TN']=tn
    df_log_mstr.at[3,'FN']=fn
    df_log_mstr.at[3,'Accuracy']=accuracy 
    df_log_mstr.at[3,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[3,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP) 
   
    #_________________________ A Gaussian SVM __________________
    
    X = df[list(df.columns.values)[1:108]].values
    Y = df[['havarth3']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)
    
    svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier.fit(X_train, Y_train.ravel())
    
    prediction = svm_classifier.predict(X_test)

    accuracy=accuracy_score(Y_test,prediction)
        
    print(" Gaussian SVM - Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)
    print("\n")    
    

    
    ## collecing in master dataframe for final summary
    df_log_mstr.at[4,'Model']='Gaussian SVM'
    df_log_mstr.at[4,'TP']=tp
    df_log_mstr.at[4,'FP']=fp
    df_log_mstr.at[4,'TN']=tn
    df_log_mstr.at[4,'FN']=fn
    df_log_mstr.at[4,'Accuracy']=accuracy 
    df_log_mstr.at[4,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[4,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)  

   #____________________ polynomial kernel SVM of degree 3 ________________________

    
    X = df[list(df.columns.values)[1:108]].values
    Y = df[['havarth3']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)
    
    svm_classifier = svm.SVC(kernel='poly', degree=3)
    svm_classifier.fit(X_train, Y_train.ravel())
    
    prediction = svm_classifier.predict(X_test)

    accuracy=accuracy_score(Y_test,prediction)
        
    print(" polynomial kernel SVM of degree 3M - Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)
    print("\n")    
    

## collecing in master dataframe for final summary
    df_log_mstr.at[5,'Model']='Polynomial Kernel SVM'
    df_log_mstr.at[5,'TP']=tp
    df_log_mstr.at[5,'FP']=fp
    df_log_mstr.at[5,'TN']=tn
    df_log_mstr.at[5,'FN']=fn
    df_log_mstr.at[5,'Accuracy']=accuracy 
    df_log_mstr.at[5,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[5,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)  

    print("Summary of Classification - note Random Forest and KNN covered seperately ")    
    print(df_log_mstr)
    print("\n")    
    
    ######################## 
    import math
    
    def calc_entropy(column):
        
        """
        Calculate entropy given a pandas series, list, or numpy array.
        """
        # Compute the counts of each unique value in the column
        counts = np.bincount(column)
        # Divide by the total column length to get a probability
        probabilities = counts / len(column)
        
        # Initialize the entropy to 0
        entropy = 0
        # Loop through the probabilities, and add each one to the total entropy
        for prob in probabilities:
            if prob > 0:
                # use log from math and set base to 2
                entropy += prob * math.log(prob, 2)
        
        return -entropy
    
    def calc_information_gain(data, split_name, target_name):
        """
        Calculate information gain given a data set, column to split on, and target
        """
        # Calculate the original entropy
        original_entropy = calc_entropy(data[target_name])
        
        #Find the unique values in the column
        values = data[split_name].unique()
        
        
        # Make two subsets of the data, based on the unique values
        left_split = data[data[split_name] == values[0]]
        right_split = data[data[split_name] == values[1]]
        
        # Loop through the splits and calculate the subset entropies
        to_subtract = 0
        for subset in [left_split, right_split]:
            prob = (subset.shape[0] / data.shape[0]) 
            to_subtract += prob * calc_entropy(subset[target_name])
        
        # Return information gain
        return original_entropy - to_subtract
    
    #columns = ['x.aidtst3', 'employ1', 'income2', 'weight2', 'height3', 'children']
    columns = df.columns

        
    dfigain = pd.DataFrame({'col' : [],
                            'infogain' : []
                        })
    
    ## Calculating the information gain in chunks
    for i in range(1,20):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])
    
    for i in range(19,25):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])
    # row 29 is eliminated due
    for i in range(24,29):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])    
    for i in range(30,40):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])
    
    for i in range(39,50):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])
    
    for i in range(49,70):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])
    
    for i in range(69,90):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])
    
    for i in range(89,107):
        dfigain.at[i,'col']=df.columns[i]
        dfigain.at[i,'infogain']=calc_information_gain(df,df.columns[i],df.columns[i])
   
    # capturing all the information gain in a dataframe  
    
    print("Round 2 - Attribute selcetion method - Information gain is developed the select the the attribtue with best info gain")
    print("\n")      
    print("This new attribute selection is going to be used for same set of classfiers and results are compared")
    print("\n")      
    print("Printing Information Gain for data set, columns are sorted by infogain descending")    
    
    
    print(dfigain[dfigain["infogain"] > 3].sort_values('infogain',ascending=False))
    
    
      
    newcol=dfigain[dfigain["infogain"] > 3]['col']
    
    print("\n")
    print("********* Round 2 of running classifiers with new attribute selection with highest information gain*******")    
   
    
   ################ Logistic Regression - Round 2 of running classifiers with new attribute selection
    
    df_log_reg = pd.DataFrame({'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })
    log_reg_classifier  = LogisticRegression(solver='lbfgs', max_iter=10000)
    #X = df[list(df.columns.values)[1:108]].values
    X= df[list(newcol)].values
    Y = df[['havarth3']].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
    log_reg_classifier.fit( X_train,Y_train.ravel())
    yprediction = log_reg_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)
      
    print("\n")    
    
    print(" Logistic Regression- Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, yprediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)
    
    
######################## Decision Tree
        
    from sklearn import tree
        
    X= df[list(newcol)].values
    Y = df[['havarth3']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
        
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf = clf.fit(X_train,Y_train)
    prediction = clf.predict(X_test)
    
    accuracy=accuracy_score(Y_test,prediction)
    
    print("\n")    
    
    print(" Decision Tree - Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)
    print("\n")    
    
    
    
    # _____________________________ naive_bayes - Round 2 of running classifiers with new attribute selection _____________________________
    
    
    from sklearn.naive_bayes import GaussianNB
        
    X= df[list(newcol)].values
    Y = df[['havarth3']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
        
    NB_classifier = GaussianNB().fit(X_train, Y_train.ravel())
    prediction = NB_classifier.predict(X_test)     
    
    accuracy=accuracy_score(Y_test,prediction)
    
    print("\n")    
    
    print(" Naive Bayes - Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)    
    
    
    
    #################### KNN - Round 2 of running classifiers with new attribute selection
    
    ### Putting the result to dataframe for easy reading    
    dfn = pd.DataFrame({'K' : [],
                        'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })
    accuracy = []
    for k in range (1,28,2):
        
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        
        X= df[list(newcol)].values
        Y = df[['havarth3']].values
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33)
        knn_classifier.fit( X_train,Y_train.ravel())
        yprediction = knn_classifier.predict(X_test)    
        accuracy=accuracy_score(Y_test,yprediction)
        tn, fp, fn, tp = confusion_matrix(Y_test, yprediction).ravel()
        
        dfn.at[k,'K']=k
        dfn.at[k,'TP']=tp
        dfn.at[k,'FP']=fp
        dfn.at[k,'TN']=tn
        dfn.at[k,'FN']=fn
        dfn.at[k,'Accuracy']=accuracy 
        dfn.at[k,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
        dfn.at[k,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    print("\n")    
    
    print(" KNN - Summary ")    
    print(dfn)
    print("\n")    
    

    ######################## 
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ')    