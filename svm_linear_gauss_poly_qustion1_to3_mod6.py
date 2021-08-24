# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 1 to Question 5
Description of Problem (just a 1-2 line summary!)
    
######### READ ME : Flow of the script #####################    
    
Please note all question are given in one script 

--> Please note Question 3.2 and up uses 2 features for all calculation , plotting and confusion matrix s

"""

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import svm
from sklearn.linear_model import LogisticRegression

url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset'
data = pd.read_csv(url,delim_whitespace=True)#
df = pd.DataFrame(data)
df.columns=['f1','f2','f3','f4','f5','f6','f7','class']

# BU ID 9%3 - remainder = 0 
# R = 0: class L = 1 (negative) and L = 2 (positive)
    
dfn=df[df["class"].isin([1,2])]
np.random.seed(10001)


try:   
    
            
    ## Quesion 1
    print("\n")
    print("Question  1 part 1")
    print("\n")
    print(df.head(5))
    print("\n")
    
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
    
    df_log_reg = pd.DataFrame({'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })    
    
    #__________________________________________________________
    #____________________ linear kernel SVM
    
    
    X = dfn[['f1','f2','f3','f4','f5','f6','f7']].values
    Y = dfn[['class']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)
    
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(X_train, Y_train.ravel())
    
    prediction = svm_classifier.predict(X_test)
    #accuracy = svm_classifier.score (X,Y)
    
    print(" linear kernel SVM - Summary ")    
    print("\n")
    print("Question 1 - part 1 - accuracy")    
    accuracy=accuracy_score(Y_test,prediction)
    print("\n")    
    print("accuracy linear kernel SVM :",round(accuracy,6))    
    print("\n")  
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    print("Question 1 part 1 - linear kernel SVM - Summary")    
    print("\n")
    print(df_log_reg)  
    
    ## collecing in master dataframe for final summary
    df_log_mstr.at[0,'Model']='linear SVM'
    df_log_mstr.at[0,'TP']=tp
    df_log_mstr.at[0,'FP']=fp
    df_log_mstr.at[0,'TN']=tn
    df_log_mstr.at[0,'FN']=fn
    df_log_mstr.at[0,'Accuracy']=accuracy 
    df_log_mstr.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    
    
    #______________________________________________________________________
    
    #____________________ A Gaussian SVM
    
    
    X = dfn[['f1','f2','f3','f4','f5','f6','f7']].values
    Y = dfn[['class']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)
    
    svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier.fit(X_train, Y_train.ravel())
    
    prediction = svm_classifier.predict(X_test)
    #accuracy = svm_classifier.score (X,Y)
    
    print("\n")
    print(" Gaussian SVM - Summary ")    
    print("\n")
    print("Question 1 - part 2 - accuracy")    
    accuracy=accuracy_score(Y_test,prediction)
    print("\n")    
    print("accuracy Gaussian SVM :",round(accuracy,6))    
    print("\n")  
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print("Question 1 part 2 - Gaussian SVM - Summary")    
    print("\n")
    print(df_log_reg)    
    
    ## collecing in master dataframe for final summary
    df_log_mstr.at[1,'Model']='Gaussian SVM'
    df_log_mstr.at[1,'TP']=tp
    df_log_mstr.at[1,'FP']=fp
    df_log_mstr.at[1,'TN']=tn
    df_log_mstr.at[1,'FN']=fn
    df_log_mstr.at[1,'Accuracy']=accuracy 
    df_log_mstr.at[1,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[1,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)                                            
    
    
    #_____________________________________________________________________
    
    #____________________ polynomial kernel SVM of degree 3
    
    X = dfn[['f1','f2','f3','f4','f5','f6','f7']].values
    Y = dfn[['class']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform (X)
    
    svm_classifier = svm.SVC(kernel='poly', degree=3)
    svm_classifier.fit(X_train, Y_train.ravel())
    
    prediction = svm_classifier.predict(X_test)
    #accuracy = svm_classifier.score (X,Y)
    
    print("\n")
    print(" polynomial kernel SVM of degree 3M - Summary ")    
    print("\n")
    print("Question 1 - part 3 - accuracy")    
    accuracy=accuracy_score(Y_test,prediction)
    print("\n")    
    print("accuracy polynomial kernel SVM of degree 3:",round(accuracy,6))    
    print("\n")  
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print("Question 1 part 3 - polynomial kernel SVM of degree 3 - Summary")    
    print("\n")
    print(df_log_reg)    
                                                
    ## collecing in master dataframe for final summary
    df_log_mstr.at[2,'Model']='polynomial SVM'
    df_log_mstr.at[2,'TP']=tp
    df_log_mstr.at[2,'FP']=fp
    df_log_mstr.at[2,'TN']=tn
    df_log_mstr.at[2,'FN']=fn
    df_log_mstr.at[2,'Accuracy']=accuracy 
    df_log_mstr.at[2,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[2,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)                                            
    
    #___________________________Question 2__________________________________________
    
    #____________________ Logistic
    
    X = dfn[['f1','f2','f3','f4','f5','f6','f7']].values
    Y = dfn[['class']].values
    
    log_reg_classifier  = LogisticRegression()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    log_reg_classifier.fit( X_train,Y_train.ravel())
    prediction = log_reg_classifier.predict(X_test)    
    
    print("\n")
    print(" Logistic Regression - Summary ")    
    print("\n")
    print("Question 2 - Logistic Regression - accuracy")    
    accuracy=accuracy_score(Y_test,prediction)
    print("\n")    
    print("accuracy Logistic Regression:",round(accuracy,6))    
    print("\n")  
    
    tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print("Question 2 Logistic Regression - Summary")    
    print("\n")
    print(df_log_reg)    
    
    ## collecing in master dataframe for final summary
    df_log_mstr.at[3,'Model']='Logistic Regression'
    df_log_mstr.at[3,'TP']=tp
    df_log_mstr.at[3,'FP']=fp
    df_log_mstr.at[3,'TN']=tn
    df_log_mstr.at[3,'FN']=fn
    df_log_mstr.at[3,'Accuracy']=accuracy 
    df_log_mstr.at[3,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_mstr.at[3,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)                                            
    
    print("\n")
    print("Question 2 - Summary")    
    print(df_log_mstr)
    print("\n")
    print("Observations-1 Logistic regression  is giving the highest acuracy ")    
    print("Observations-2 Linear SVM is giving better accuracy over other SVN classificaiton")    
    print("Observations-3 Gaussian SVM is giving the lowest of the accuracy of all classifcaiton for this paticular dataset")    
    
    
    #___________________________Question 3__________________________________________
    
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    
    # Compute and plot distortion vs k. Use the "knee" method to nd the best k.
    x = df
    inertia_list = []
    for k in range(1,9):
        kmeans_classifier = KMeans(n_clusters=k)
        y_kmeans = kmeans_classifier.fit_predict(x)
        inertia = kmeans_classifier.inertia_
        inertia_list.append(inertia)

    fig,ax = plt.subplots(1,figsize =(7,5))
    plt.plot(range(1, 9), inertia_list, marker='o',
        color='green')

    #plt.legend()
    plt.xlabel('number of clusters: k')
    plt.ylabel('inertia')
    plt.axvline(x=3, label='Best k is at k = {}'.format(3), color='b', linestyle='--')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    print("\n")
    print("THe best K is k=3 since the loss function does not have high variation after k =3")    
    print("\n")
    
    #___________________________Question 3 part 2__________________________________________
    
    # re-run your clustering with best k clusters.
    
    # Selecting randomw features fi and fj
    
    dfnew=df[df[['f1','f2','f3','f4','f5','f6','f7']].columns.to_series().sample(2)]
    
    dfnew['class']=df['class']
    x = dfnew
    # Best k =3
    kmeans_classifier = KMeans(n_clusters=3)
    y_kmeans = kmeans_classifier.fit_predict(x)
    centroids = kmeans_classifier.cluster_centers_
    cluster = kmeans_classifier.predict(x)
    clabels=kmeans_classifier.labels_
    
    xi = dfnew[dfnew.columns[0]]
    yi = dfnew[dfnew.columns[1]]
    plt.scatter(xi,yi,c=y_kmeans,s=75,cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:,1] ,s = 200 , c = 'red', label = 'centroild') 
    
    
    plt.xlabel(xi.name)
    plt.ylabel(yi.name)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    print('Question 3 part 2 - Examine your plot. Are there any interesting patterns? ')
    print("\n")
    print(" Observation - The scatter plot is attached in seperate attachment.") 
    print(" The scatter plot maps feature f3 vs f5 , Well I do not see any specific pattern for these these classes")
    
    #___________________________Question 3 part 3__________________________________________
        
    ## Adding a columns cluster to the dataframe
    dfnew['cluster']=cluster
    
    ## using group by  to find by max count
    
    c1=dfnew.groupby(['cluster','class'],as_index=False).size()  
    c2=c1.groupby(['cluster'],as_index=False).max()
    
    #c1=dfnew.groupby(['cluster','class'],as_index=False).count()
    
    print('Question 3 part 3 ')
    
    df_log_reg1=c2[['cluster','class']]
    df_log_reg1['centroid']=[centroids[0][[0,1]],centroids[1][[0,1]],centroids[2][[0,1]]]
    
    print("\n")
    print(df_log_reg1)   

    #___________________________Question 3 part 4__________________________________________


    dfnew['newclass']=''
    
    for i in range(len(dfnew)):
        
        x=dfnew.loc[i][0]
        y=dfnew.loc[i][1]
        
        # cluster 0
        cen_xa=df_log_reg1.loc[0][2][0]
        cen_ya=df_log_reg1.loc[0][2][1]
        # cluster 1
        cen_xb=df_log_reg1.loc[1][2][0]
        cen_yb=df_log_reg1.loc[1][2][1]
        # cluster 2
        cen_xc=df_log_reg1.loc[2][2][0]
        cen_yc=df_log_reg1.loc[2][2][1]
        
        # calcualte Euc distance 
        euc_dist_a=round(np.sqrt((x-cen_xa)*(x-cen_xa)+(y-cen_ya)*(y-cen_ya)),4)
        euc_dist_b=round(np.sqrt((x-cen_xb)*(x-cen_xb)+(y-cen_yb)*(y-cen_yb)),4)
        euc_dist_c=round(np.sqrt((x-cen_xc)*(x-cen_xc)+(y-cen_yc)*(y-cen_yc)),4)
        
        temp_list=[euc_dist_a,euc_dist_b,euc_dist_c]
        ## finding the closet cluster, minimum eucd dist
        vartemp=temp_list.index(min(temp_list))
        #print(temp_list,i,vartemp)
        if vartemp==0:
            dfnew.at[i,'newclass']=2
        elif vartemp==1:  
            dfnew.at[i,'newclass']=3
        elif vartemp==2:  
            dfnew.at[i,'newclass']=1    
        else:
            pass
    
    #print(dfnew.head(5))
    
    print("\n")
    print('Question 3 part 4 - overall accuracy of this new classier ')
    print("\n")
    ## accuracy of new classifier
    acc_new_classfier=round(len(dfnew.loc[(dfnew["class"] == dfnew["newclass"])])/len(dfnew),5)
    
    print("accuracy of new classifier = ",acc_new_classfier)
    print("\n")
    print('Observation - accuracy of 82% with manual classifier is fairly good')
    
   
    #___________________________Question 3 part 4__________________________________________

   
    # BU ID 9%3 - remainder = 0 
    # R = 0: class L = 1 (negative) and L = 2 (positive)
    
    dfq3part4=dfnew[dfnew["class"].isin([1,2])]
    
    print("\n")
    print('Question 3 part 5 ')
    print("\n")
    ## accuracy of new classifier with L1 and L2
    acc_new_classfier=round(len(dfq3part4.loc[(dfq3part4["class"] == dfq3part4["newclass"])])/len(dfq3part4),5)
    
    print("accuracy of new classifier with L=1 and L =2 : ",acc_new_classfier)
    print("\n")
    print(" Observation 1 - The accuracy of the classifer decreases with only two attributes ")
    print(" Observation 2 - Also classifier with only two attribtues is giving the lowest accuracy comparaed to all classifiers in question 2")
        
except Exception as e:
    print(e)
    print(' Thsere is some issue in processing, please check ')    

