# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 1 to 6
Description of Problem (just a 1-2 line summary!): Quesion 1 to Question 6 


Please note - All problems are given in the same script 
   
"""
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
data = pd.read_csv(url, names=['f1', 'f2','f3', 'f4', 'Class'],delimiter=',')
df = pd.DataFrame(data)

np.random.seed(10001)

try:   
    
    ### Question  1 part 1
    
       
    df['color'] = np.where(df["Class"] == 0,'green','red')
    
    print("Question  1 part 1")
    print("\n")
    print(df.head(5))
    print("\n")
    
    ### question 1 part 2
    
    
    dfn = pd.DataFrame({'Class' : ['0', '1', 'all'],
                        'mu(f1)':[round(statistics.mean(df[(df["Class"]== 0) ]['f1']),2),
                                  round(statistics.mean(df[(df["Class"]== 1) ]['f1']),2),
                                  round(statistics.mean(df['f1']),2)],
                        'sd(f1)':[round(statistics.stdev(df[(df["Class"]== 0) ]['f1']),2),
                                     round(statistics.stdev(df[(df["Class"]== 0) ]['f1']),2),
                                     round(statistics.stdev(df['f1']),2)],
                        'mu(f2)':[round(statistics.mean(df[(df["Class"]== 0) ]['f2']),2),
                                  round(statistics.mean(df[(df["Class"]== 1) ]['f2']),2),
                                  round(statistics.mean(df['f2']),2)],
                        'sd(f2)':[round(statistics.stdev(df[(df["Class"]== 0) ]['f2']),2),
                                     round(statistics.stdev(df[(df["Class"]== 0) ]['f2']),2),
                                     round(statistics.stdev(df['f2']),2)],
                        'mu(f3)':[round(statistics.mean(df[(df["Class"]== 0) ]['f3']),2),
                                  round(statistics.mean(df[(df["Class"]== 1) ]['f3']),2),
                                  round(statistics.mean(df['f3']),2)],
                        'sd(f3)':[round(statistics.stdev(df[(df["Class"]== 0) ]['f3']),2),
                                     round(statistics.stdev(df[(df["Class"]== 0) ]['f3']),2),
                                     round(statistics.stdev(df['f3']),2)],
                        'mu(f4)':[round(statistics.mean(df[(df["Class"]== 0) ]['f4']),2),
                                  round(statistics.mean(df[(df["Class"]== 1) ]['f4']),2),
                                  round(statistics.mean(df['f4']),2)],
                        'sd(f4)':[round(statistics.stdev(df[(df["Class"]== 0) ]['f4']),2),
                                     round(statistics.stdev(df[(df["Class"]== 0) ]['f4']),2),
                                     round(statistics.stdev(df['f4']),2)]
                       })
    
    
    print("Question 1 part 2 - Summary table ")    
    print("\n")
    print(dfn)      
    print("\n")
    
    print("Question 1 part 3 - Observation ")
    print("\n")
    print("Observation 1 - Real Notes or class 0 have postive mean for f1 while it negative for fake notes")
    print("Observation 2 - For f1 THe stddev for real and  fake notes are same")
    print("Observation 3 - Real Notes or class 0 have postive mean for f2 while it negative for fake notes")
    print("Observation 4 - Real Notes have a lower mean for f3 vs fake notes")
    print("Observation 5 - Real Notes and fake notes have same mean and std deviation")
    print("\n")
    
    ################### quesiont 2
    
    ## part 1 split the data 50:50 manully without using the test train split    
    
    X = df[['f1','f2','f3','f4']].values
    Y = df[['color']].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    
    df_train = np.concatenate((X_train, Y_train), axis=1)
    df_train=pd.DataFrame(df_train)
    df_train.columns = ['f1', 'f2','f3', 'f4','Class']
    
    df_test = np.concatenate((X_test, Y_test), axis=1)
    df_test=pd.DataFrame(df_test)
    df_test.columns = ['f1', 'f2','f3', 'f4','Class']
     
    ## plot X Train Class 0    
    fig_trian_class0 = sns.pairplot(df_train[df_train["Class"]== 'green'][["f1","f2","f3","f4","Class"]],hue=None)
    plt.show()
    fig_trian_class0.savefig("Xtrain_class0.pdf", bbox_inches='tight')
    
        
    ## plot X Train Class 1
    fig_trian_class1 = sns.pairplot(df_train[df_train["Class"]== 'red'][["f1","f2","f3","f4","Class"]])
    plt.show()
    fig_trian_class1.savefig("Xtrain_class1.pdf", bbox_inches='tight')
    
    
    ### creating own classifier logic
    
    ####################

    ## Question 2 part 3
    
    df_test.reset_index(drop=True,inplace=True)
    for i in  range(len(df_test)):
        
        # Creatig logic for simple classifier
        if (df_test.loc[i,'f1'] >= -2 and  df_test.loc[i,'f1'] <= 5) and (df_test.loc[i,'f2'] >=-9 and df_test.loc[i,'f2'] <= 10.5 ) and (df_test.loc[i,'f3'] >=-5 and df_test.loc[i,'f3'] <= 6 ) and (df_test.loc[i,'f4'] >=-5 and df_test.loc[i,'f4'] <= 0.5 ):
            
            
            df_test.at[i,'Predict']='green'
        else:
            df_test.at[i,'Predict']='red'
    print("\n")
    print("Question 2 part 3")
    print("\n")
    print(df_test.head(5))
    
    
    ## Question 2 part 4
    
    ### Question 2 - Part 4 - True Positive true positives (your predicted label is + and true labelis +
    df_test_tp_count=df_test[(df_test["Class"]=='green') & (df_test["Predict"]=='green')]['Predict'].count()
    print("\n")
    print("Question 2 Part 4(a)  - TP for simple classifier is ", df_test_tp_count)
    
    ### Question 2 - Part 4 - FP - false positives (your predicted label is + but true labelis -
    df_test_fp_count=df_test[(df_test["Class"]=='red') & (df_test["Predict"]=='green')]['Predict'].count()
    print("\n")
    print("Question 2 Part 4(b)  - FP for simple classifier is ", df_test_fp_count)
    
    ### Question 2 - Part 4 - TN - true negativess (your predicted label is - and truelabel is -
    df_test_tn_count=df_test[(df_test["Class"]=='red') & (df_test["Predict"]=='red')]['Predict'].count()
    print("\n")
    print("Question 2 Part 4(c)  - TN for simple classifier is ", df_test_tn_count)
    
    ### Question 2 - Part 4 - FN - false negatives (your predicted label is - but true label is +
    df_test_fn_count=df_test[(df_test["Class"]=='green') & (df_test["Predict"]=='red')]['Predict'].count()
    
    print("\n")
    print("Question 2 Part 4(d)  - FN for simple classifier is ", df_test_fn_count)
    
    ### Question 2 - Part 4 - TPR = TP/(TP + FN) - true positive rate.
    df_test_tpr_count= round(df_test_tp_count/ ( df_test_tp_count+df_test_fn_count),3)
    print("\n")
    print("Question 2 Part 4(e)  - TPR for simple classifier is ", df_test_tpr_count)
    
    ### Question 2 - Part 4 - TNR = TN/(TN + FP) - true negative rate.
    df_test_tnr_count= round(df_test_tn_count/ ( df_test_tn_count+df_test_fp_count),3)
    
    print("\n")
    print("Question 2 Part 4(f)  - TNR for simple classifier is ", df_test_tnr_count)
    
    ### Question 2 - Calclating Accuracy = (TP + TN)/All
    df_test_accuracy_count= round(( df_test_tp_count + df_test_tn_count ) / len(df_test),3)
    
     
    dfn = pd.DataFrame({'TP':[df_test_tp_count],
                       'FP':[df_test_fp_count],
                       'TN':[df_test_tn_count],
                       'FN':[df_test_fn_count],
                       'Accuracy':[df_test_accuracy_count],
                       'TPR':[df_test_tpr_count],
                       'TNR':[df_test_tnr_count]
                       })
    
    print("\n")
    
    print("Question 2  - Part 5 - Summary ")
    print("\n")
    ## Printing the summary output - Question 4 - Part 4 - Summary
    print(dfn) 
    
    print("\n")
    print("Question 2  - Part 6 : The simple classifier is performing better than Coin flip as it is giving accuracy of 63% which is good for a manually made classifier")
    print("\n")
    
    
    feature_names = ['f1','f2','f3','f4']
    X = df[['f1','f2','f3','f4']].values
    Y = df[['color']].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    

    knn_classifier = KNeighborsClassifier(n_neighbors=5)  
    
    knn_classifier.fit( X_train,Y_train.ravel())#.ravel()
    yprediction = knn_classifier.predict(X_test)    
    
    cm = confusion_matrix(Y_test, yprediction)
    tn, fp, fn, tp = confusion_matrix(Y_test, yprediction).ravel()
    
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
    for k in range (1,21,2):
        #print(k)
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
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
    
    
    ################ Question 3 - part 2 and part 3
    
    print("\n")    
    print("Question 3 - part 1 & 3 ")    
    print("\n")   
    
    print(dfn) 
    print("\n")   
    print("Question 3 - part 2 - Printing the plot")    
    dfn.plot(x='K',y='Accuracy')
    ################ Question 3 - part 4
    print("\n")    
    print("Question 3 - part 4")    
    print("\n")   
    print("Comments - KNN is better than the simplified classifier as the accuracy of the model is higher than the simpler classifier. All paraters for the KNN are better than the simplier model")    
    
    ################ Question 3 - part 5
    
    print("\n")    
    print("Question 3 - part 5 - with BUID as input , last 4 digit")    
    # BU ID U43598696
    x_user=[[8,6,9,6]]
    knn_classifier = KNeighborsClassifier(n_neighbors=11)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    knn_classifier.fit( X_train,Y_train.ravel())
    
    yprediction = knn_classifier.predict(x_user)
    print("yprediction",yprediction)
    print("\n")    
    print("Comments - Looks like the with User input as BU ID, the model has prediceted as legitimate bank notes ")
    
    
    ################ Question 4 - part 1
    
    print("\n")    
    print("Question 4 - part 1 (1) - with  f1 missing")    
    
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f1
    X_train=np.delete(X_train, 0, 1)
    X_test=np.delete(X_test, 0, 1)
    knn_classifier.fit( X_train,Y_train.ravel())
    yprediction = knn_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)
    print("accuracy=",accuracy)
    
    
    print("\n")    
    print("Question 4 - part 1 (2) - with  f2 missing")    
    
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f2
    X_train=np.delete(X_train, 1, 1)
    X_test=np.delete(X_test, 1, 1)
    knn_classifier.fit( X_train,Y_train.ravel())
    yprediction = knn_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)
    print("accuracy=",accuracy)
    
    print("\n")    
    print("Question 4 - part 1 (3) - with  f3 missing")    
    
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f3
    X_train=np.delete(X_train, 2, 1)
    X_test=np.delete(X_test, 2, 1)
    knn_classifier.fit( X_train,Y_train.ravel())
    yprediction = knn_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)
    print("accuracy=",accuracy)
    
    print("\n")    
    print("Question 4 - part 1 (4) - with  f4 missing")    
    
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f4
    X_train=np.delete(X_train, 3, 1)
    X_test=np.delete(X_test, 3, 1)
    knn_classifier.fit( X_train,Y_train.ravel())
    yprediction = knn_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)
    print("accuracy=",accuracy)
    
    ###################################
    
    print("\n")    
    print("Question 4 - part 2")    
       
    print("Comments - Accuracy did not increase in any of the cases when there was missing feature vs when all the featured where considered")    
    print("\n")    
    print("Question 4 - part 3 - which feature, when removed, contributed the most to loss of accuracy")    
    
    print("Comments - The feature when removed that contributed to most loss was f1 , as accuracy did drop by around 6 %  ")
    
    print("\n")    
    print("Question 4 - part 4 - which feature, when removed, contributed the least to loss of accuracy?")    
    
    print("Comments - The feature when removed which contriuted to leaset loss was f4 ,as the loss of accuracy was less than 0.01% indicating that there was no gain in having this feature in the model")
    print("\n")    
    
    ################ Question 5 ###############
    
    ################ Question 5 part 1
    
    print("Question 5 - part 1 - Logistic Regression ")    
    
    df_log_reg = pd.DataFrame({'TP' : [],
                        'FP' : [],
                        'TN' : [],
                        'FN' : [],
                        'Accuracy' : [],
                        'TPR' : [],
                        'TNR' : [],
                        })
    log_reg_classifier  = LogisticRegression()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    log_reg_classifier.fit( X_train,Y_train.ravel())
    yprediction = log_reg_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)
    print("Logistic Regression accuracy=",accuracy)
      
    print("\n")    
    
    print("Question 5 - part 2 - Logistic Regression- Summary ")    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, yprediction).ravel()
    
    df_log_reg.at[0,'TP']=tp
    df_log_reg.at[0,'FP']=fp
    df_log_reg.at[0,'TN']=tn
    df_log_reg.at[0,'FN']=fn
    df_log_reg.at[0,'Accuracy']=accuracy 
    df_log_reg.at[0,'TPR']=round(tp/(tp+fn),3) # TPR = TP/(TP + FN)
    df_log_reg.at[0,'TNR']=round(tn/(tn+fp),3) # TNR = TN/(TN + FP)
    print(df_log_reg)
    print("\n")    
        
    print("Question 5 - part 3")    
    print("Comments - logistic regression is definately better than your simple classifer since the simple classifer is giving 60% accuracy while logistic regerssion is giving around 99% accuracy")    
    print("\n")    
    
    print("Question 5 - part 4")    
    print("Comments - Well in this particular case logistic regression is beaten marginally by knn but again this is only for this small data set ")    
    print("\n")    
    
    print("Question 5 - part 5 - BUID prediction Logistic regression ")    
    # BU ID U43598696
    x_user=[[8,6,9,6]]
    log_reg_classifier  = LogisticRegression()
    log_reg_classifier.fit( X_train,Y_train.ravel())
    yprediction = log_reg_classifier.predict(x_user)
    print("yprediction",yprediction)
    print("\n")    
    print("Comments - Looks like the with User input as BU ID, the model has prediceted as legitimate bank notes ")
    
    ################ Question 6 ###############
    
    ################ Question 6 part 1
    print("\n")    
    print("Question 6 - part 1 (1) - with  f1 missing")    
    
    log_reg_classifier  = LogisticRegression()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f1
    X_train=np.delete(X_train, 0, 1)
    X_test=np.delete(X_test, 0, 1)
    log_reg_classifier.fit( X_train,Y_train.ravel())    
    yprediction = log_reg_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)    
    print("LogisticRegression accuracy=",accuracy)
    
    print("\n")    
    print("Question 6 - part 1 (2) - with  f2 missing")    
    
    log_reg_classifier  = LogisticRegression()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f2
    X_train=np.delete(X_train, 1, 1)
    X_test=np.delete(X_test, 1, 1)
    log_reg_classifier.fit( X_train,Y_train.ravel())    
    yprediction = log_reg_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)    
    print("LogisticRegression accuracy=",accuracy)
    
    print("\n")    
    print("Question 6 - part 1 (3) - with  f3 missing")    
    
    log_reg_classifier  = LogisticRegression()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f3
    X_train=np.delete(X_train, 2, 1)
    X_test=np.delete(X_test, 2, 1)
    log_reg_classifier.fit( X_train,Y_train.ravel())    
    yprediction = log_reg_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)    
    print("LogisticRegression accuracy=",accuracy)
    
    print("\n")    
    print("Question 6 - part 1 (4) - with  f4 missing")    
    
    log_reg_classifier  = LogisticRegression()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
    # dropping f4
    X_train=np.delete(X_train, 3, 1)
    X_test=np.delete(X_test, 3, 1)
    log_reg_classifier.fit( X_train,Y_train.ravel())    
    yprediction = log_reg_classifier.predict(X_test)    
    accuracy=accuracy_score(Y_test,yprediction)    
    print("LogisticRegression accuracy=",accuracy)
    
    #################
    
    print("\n")    
    print("Question 6 - part 2")    
    print("Comments - Accuracy decreased for the cases for missing f1 to f4 while for features missing f4 the accuracy is almost same down to fifth decimal digit")
    
    print("\n")    
    print("Question 6 - part 3 - which feature, when removed, contributed the most to lossof accuracy?")    
    print("Comments - The feature when removed that contributed to most loss was f1 , as accuracy did drop from 99% to 80% ")
    
    print("\n")    
    print("Question 6 - part 4 - which feature, when removed, contributed the least to loss of accuracy?")    
    print("Comments - The feature when removed which contributed to least loss was f4 ,as the loss of accuracy was less than 1% indicating that there was no gain in having this feature in the model")
    
    print("\n")    
    print("Question 6 - part 5 - is relative significance of features the same as you obtained using k-NN?")    
    print("Comments - So significance of feature which is most important is f1 and least important is f4 for both knn and logistic which indicate that significance of features are same ")
    
    
except Exception as e:
    print(e)
    print('failed to excute correctly')