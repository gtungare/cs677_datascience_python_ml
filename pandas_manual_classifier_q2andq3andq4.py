# -*- coding: utf-8 -*-
"""
Name - Gaurav Tungare
Class: CS 677 - Summer 2
Date: 7/7/2021
Homework  # Question 2, 3 & 4
Description of Problem (just a 1-2 line summary!): Buy and Hold strategy

######### READ ME : Flow of the script #####################    
    
Ticker Used - TM , Duration used - 2015 to 2019 
Ticker Used - SPY , Duration used - 2015 to 2019 

Training Set - 2015, 2016 , 2017
Test Set - 2018,2019

Step 1 : CSV file are read and created to a list , one for TM ticker and other for SPY ticker
Step 2 : For each ticker , Trube label is calcualted
Step 3 : For each ticker ,For loop is used to calculate the probability
   
"""
import os
import csv
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import pickle

ticker='TM'
# Second ticker
ticker2='SPY'
here =os.path.abspath( __file__ )
input_dir =os.path.abspath(os.path.join(here ,os. pardir ))
ticker_file = os.path.join(input_dir, ticker + '.csv')
ticker_file_spy = os.path.join(input_dir, ticker2 + '.csv')

try:   
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)
    print('The script takes few mins to run, please be patient  ')
    """    Code for Question 2 to Question 4
    """
        
    
    # using uft encoding 
    # Casting the line of xls to a list 
    data= open(ticker_file,encoding='utf-8')
    csv_data = csv.reader(data, delimiter=',')
    
    data_line= list(csv_data)
    data_tickertm = pd.DataFrame(data_line)
    
    # Reading using pandas
    df=pd.read_csv(ticker_file) 
        
    
    ## Reading the SPY ticker
    data_spy= open(ticker_file_spy,encoding='utf-8')
    csv_data_spy = csv.reader(data_spy, delimiter=',')
    data_line_spy= list(csv_data_spy)
    
    # Reading using pandas
    df_spy=pd.read_csv(ticker_file_spy) 
    
    ## Creating a new column called TrueLabel 
    ## If the return is greater than 0 then "Plus" else "Neg"
    
    df['TrueLabel'] = np.where(df["Return"] > 0,'+','-')
    
    # Splitting between test and train set for TM Ticker
    
    ## Training set for TM Ticker - 2015,2016,2017
    df_year123 = df[df["Year"].isin([2015,2016,2017])]
    
    ## Test set for TM Ticker - Year 2018 and 2019
    df_year45 = df[df["Year"].isin([2018,2019])]

    ## Creating a new column called TrueLabel for SPY ticker
    ## If the return is greater than 0 then "Plus" else "Neg"
    
    df_spy['TrueLabel'] = np.where(df_spy["Return"] > 0,'+','-')
    
    # Splitting between test and train set for TM Ticker
    
    ## Training set for SPY Ticker - 2015,2016,2017
    df_spy_year123 = df_spy[df_spy["Year"].isin([2015,2016,2017])]
    
    ## Test set for SPY Ticker - Year 2018 and 2019
    df_spy_year45 = df_spy[df_spy["Year"].isin([2018,2019])]

    
    ## for TM
   
    nsplus=0
    nsminus=0
    s_minus='-'
    s_plus='+'
    ## Index is reset to start from 0 
    
    
    df_year123.reset_index(drop=True,inplace=True)    
    df_year45.reset_index(drop=True,inplace=True)
    
    # 
    # list goes till the second last row in the training set 
    ## calcuation for W = 2
    ## Checking for current and current-1 labels from the test set
    
    # For hyperparameter -  W = 2
    
    df_year45['W2']=''
    b=0
    for a in range((len(df_year45)-1)):
        nsminus=0
        nsplus=0
        b=a+1
        s_label=df_year45.loc[b-1,"TrueLabel"]+df_year45.loc[b,"TrueLabel"]
        #s_plus_train=df_year45.loc[a,"TrueLabel"]+df_year45.loc[a+1,"TrueLabel"]+df_year45.loc[a+2,"TrueLabel"]
        #print(s_label)
        for i in range((len(df_year123)-2)):
            # Compare the train label w/ + with test label
            if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]==s_label+s_minus:
                nsplus = nsplus+1
            # Compare the train label w/ - with test label
            elif df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]==s_label+s_plus:
                nsminus = nsminus+1
            else:
                pass
        # THis step for default probability
        if a==0:
            df_year45.at[a,'W2']='+'
        else:
            pass
        # applying the logic for If N+(s) <N-(s) then the next day is assigned "+". If N+(s) < N-(s) then thenext day is assigned "-".
        
        if nsplus>nsminus:
            df_year45.at[b,'W2']='+'
        else:
            df_year45.at[b,'W2']='-'
        
    
    # For hyperparameter -  W = 3
    df_year45['W3']=''
    b=0
    for a in range((len(df_year45)-2)):
        nsminus=0
        nsplus=0
        b=a+2
        s_label=df_year45.loc[b-2,"TrueLabel"]+df_year45.loc[b-1,"TrueLabel"]+df_year45.loc[b,"TrueLabel"]
        #s_plus_train=df_year45.loc[a,"TrueLabel"]+df_year45.loc[a+1,"TrueLabel"]+df_year45.loc[a+2,"TrueLabel"]
        #print(s_label)
        for i in range((len(df_year123)-3)):
            # Compare the train label w/ + with test label
            if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]+df_year123.loc[i+3,"TrueLabel"]==s_label+s_minus:
                nsplus = nsplus+1
            # Compare the train label w/ - with test label
            elif df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]+df_year123.loc[i+3,"TrueLabel"]==s_label+s_plus:
                nsminus = nsminus+1
            else:
                pass
        ## THis step for initial value since the prediction happens from 3rd postion 
        ## this is default probability
        if a==0 or a==1:
            df_year45.at[a,'W3']='+'
        else:
            pass
        # applying the logic for If N+(s) <N-(s) then the next day is assigned "+". If N+(s) < N-(s) then thenext day is assigned "-".
        
        if nsplus>nsminus:
            df_year45.at[b,'W3']='+'
        elif nsplus<nsminus:
            df_year45.at[b,'W3']='-'
        else:
            # assiging default 
            df_year45.at[b,'W3']='+'
        
    
    # For hyperparameter -  W = 4
    df_year45['W4']=''
    b=0
    for a in range((len(df_year45)-3)):
        nsminus=0
        nsplus=0
        b=a+3
        s_label=df_year45.loc[b-3,"TrueLabel"]+df_year45.loc[b-2,"TrueLabel"]+df_year45.loc[b-1,"TrueLabel"]+df_year45.loc[b,"TrueLabel"]
        
        for i in range((len(df_year123)-4)):
            # Compare the train label w/ + with test label
            if df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]+df_year123.loc[i+3,"TrueLabel"]+df_year123.loc[i+4,"TrueLabel"]==s_label+s_minus:
                nsplus = nsplus+1
            # Compare the train label w/ - with test label
            elif df_year123.loc[i,"TrueLabel"]+df_year123.loc[i+1,"TrueLabel"]+df_year123.loc[i+2,"TrueLabel"]+df_year123.loc[i+3,"TrueLabel"]+df_year123.loc[i+4,"TrueLabel"]==s_label+s_plus:
                nsminus = nsminus+1
            else:
                pass
        ## THis step for initial value since the prediction happens from 3rd postion 
        # assigning default 
        if a==0 or a==1 or a==2:
            df_year45.at[a,'W4']='+'
        else:
            pass
        # applying the logic for If N+(s) <N-(s) then the next day is assigned "+". If N+(s) < N-(s) then thenext day is assigned "-".
        
        if nsplus>nsminus:
            df_year45.at[b,'W4']='+'
        elif nsplus<nsminus:
            df_year45.at[b,'W4']='-'
        else:
            # default
            df_year45.at[b,'W4']='+'
               
    
    ############## FOr SPY Ticker #######################
    
    df_spy_year123.reset_index(drop=True,inplace=True)
    
    df_spy_year45.reset_index(drop=True,inplace=True)
    
    ## calcuation for W = 2
    ## Checking for current and current-1 labels from the test set
    
    # For hyperparameter -  W = 2
    df_spy_year45['W2']=''
    b=0
    for a in range((len(df_spy_year45)-1)):
        nsminus=0
        nsplus=0
        b=a+1
        s_label=df_spy_year45.loc[b-1,"TrueLabel"]+df_spy_year45.loc[b,"TrueLabel"]
        for i in range((len(df_spy_year123)-2)):
            # Compare the train label w/ + with test label
            if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]==s_label+s_minus:
                nsplus = nsplus+1
            # Compare the train label w/ - with test label
            elif df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]==s_label+s_plus:
                nsminus = nsminus+1
            else:
                pass
        # THis step for initial value 
        # Default probability
        if a==0:
            df_spy_year45.at[a,'W2']='+'
        else:
            pass
        # applying the logic for If N+(s) <N-(s) then the next day is assigned "+". If N+(s) < N-(s) then thenext day is assigned "-".
        
        if nsplus>nsminus:
            df_spy_year45.at[b,'W2']='+'
        elif nsplus<nsminus:
            df_spy_year45.at[b,'W2']='-'
        else:
            df_spy_year45.at[b,'W2']='+'
   
        
    # For hyperparameter -  W = 3
    df_spy_year45['W3']=''
    b=0
    for a in range((len(df_spy_year45)-2)):
        nsminus=0
        nsplus=0
        b=a+2
        s_label=df_spy_year45.loc[b-2,"TrueLabel"]+df_spy_year45.loc[b-1,"TrueLabel"]+df_spy_year45.loc[b,"TrueLabel"]
        
        for i in range((len(df_spy_year123)-3)):
            # Compare the train label w/ + with test label
            if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]+df_spy_year123.loc[i+3,"TrueLabel"]==s_label+s_minus:
                nsplus = nsplus+1
            # Compare the train label w/ - with test label
            elif df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]+df_spy_year123.loc[i+3,"TrueLabel"]==s_label+s_plus:
                nsminus = nsminus+1
            else:
                pass
        ## THis step for initial value since the prediction happens from 3rd postion 
        # default probability
        if a==0 or a==1:
            df_spy_year45.at[a,'W3']='+'
        else:
            pass
        # applying the logic for If N+(s) <N-(s) then the next day is assigned "+". If N+(s) < N-(s) then thenext day is assigned "-".
        
        if nsplus>nsminus:
            df_spy_year45.at[b,'W3']='+'
        elif nsplus<nsminus:
            df_spy_year45.at[b,'W3']='-'
        else:
            df_spy_year45.at[b,'W3']='+'
        #print('nsminus',nsminus,'nsplus',nsplus)   
    
    # For hyperparameter -  W = 4
    df_spy_year45['W4']=''
    b=0
    for a in range((len(df_spy_year45)-3)):
        nsminus=0
        nsplus=0
        b=a+3
        s_label=df_spy_year45.loc[b-3,"TrueLabel"]+df_spy_year45.loc[b-2,"TrueLabel"]+df_spy_year45.loc[b-1,"TrueLabel"]+df_spy_year45.loc[b,"TrueLabel"]
        
        for i in range((len(df_spy_year123)-4)):
            # Compare the train label w/ + with test label
            if df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]+df_spy_year123.loc[i+3,"TrueLabel"]+df_spy_year123.loc[i+4,"TrueLabel"]==s_label+s_minus:
                nsplus = nsplus+1
            # Compare the train label w/ - with test label
            elif df_spy_year123.loc[i,"TrueLabel"]+df_spy_year123.loc[i+1,"TrueLabel"]+df_spy_year123.loc[i+2,"TrueLabel"]+df_spy_year123.loc[i+3,"TrueLabel"]+df_spy_year123.loc[i+4,"TrueLabel"]==s_label+s_plus:
                nsminus = nsminus+1
            else:
                pass
            
        ## THis step for initial value since the prediction happens from 3rd postion 
        if a==0 or a==1 or a==2:
            df_spy_year45.at[a,'W4']='#'
        else:
            pass
        # applying the logic for If N+(s) <N-(s) then the next day is assigned "+". If N+(s) < N-(s) then thenext day is assigned "-".
        
        if nsplus>nsminus:
            df_spy_year45.at[b,'W4']='+'
        elif nsplus<nsminus:
            df_spy_year45.at[b,'W4']='-'
        else:
            df_spy_year45.at[b,'W4']='+'
    

    ############### question 2 - part 2 and 3
    df_year45.reset_index(drop=True,inplace=True)
    # for Percentage of True Vs Predicted label calculation 
    ##Initialize the count for 
    count_spy_w2=0
    count_spy_w3=0
    count_spy_w4=0
    total_spy_count=len(df_spy_year45)
    for i in range((len(df_spy_year45))):
        if df_spy_year45.loc[i,"TrueLabel"]==df_spy_year45.loc[i,"W2"]:
            count_spy_w2=count_spy_w2+1
        else:
            pass
        
        if df_spy_year45.loc[i,"TrueLabel"]==df_spy_year45.loc[i,"W3"]:
            count_spy_w3=count_spy_w3+1
        else:
            pass
        if df_spy_year45.loc[i,"TrueLabel"]==df_spy_year45.loc[i,"W4"]:
            count_spy_w4=count_spy_w4+1    
        else:
            pass
        

########################### 

    df_year45.reset_index(drop=True,inplace=True)
    # for Percentage of True Vs Predicted label calculation 
    ##Initialize the count for 
    count_w2=0
    count_w3=0
    count_w4=0
    total_count=len(df_year45)
    for i in range((len(df_year45))):
        # count True Lable Vs Predicted labe for W=2
        if df_year45.loc[i,"TrueLabel"]==df_year45.loc[i,"W2"]:
            count_w2=count_w2+1
        else:
            pass
        # count True Lable Vs Predicted labe for W=3
        if df_year45.loc[i,"TrueLabel"]==df_year45.loc[i,"W3"]:
            count_w3=count_w3+1
        else:
            pass
        # count True Lable Vs Predicted labe for W=4
        if df_year45.loc[i,"TrueLabel"]==df_year45.loc[i,"W4"]:
            count_w4=count_w4+1    
        else:
            pass
    
    w2_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W2"]=='+')]['W2'].count()
    w3_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W3"]=='+')]['W3'].count()
    w4_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W4"]=='+')]['W4'].count()    
    
    w2_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W2"]=='+')]['W2'].count()
    w3_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W3"]=='+')]['W3'].count()
    w4_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W4"]=='+')]['W4'].count()
    
    w2_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W2"]=='-')]['W2'].count()
    w3_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W3"]=='-')]['W3'].count()
    w4_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W4"]=='-')]['W4'].count()
    
    w2_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W2"]=='-')]['W2'].count()
    w3_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W3"]=='-')]['W3'].count()
    w4_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W4"]=='-')]['W4'].count()
    
    w2_tm_accuracy_count= ( w2_tm_tp_count + w2_tm_tn_count ) / len(df_year45)
    w3_tm_accuracy_count= ( w3_tm_tp_count + w3_tm_tn_count ) / len(df_year45)
    w4_tm_accuracy_count= ( w4_tm_tp_count + w4_tm_tn_count ) / len(df_year45)
    
    w2_spy_accuracy_count= ( w2_spy_tp_count + w2_spy_tn_count ) / len(df_spy_year45)
    w3_spy_accuracy_count= ( w3_spy_tp_count + w3_spy_tn_count ) / len(df_spy_year45)
    w4_spy_accuracy_count= ( w4_spy_tp_count + w4_spy_tn_count ) / len(df_spy_year45)
    
    
    print("\n")
    print("Question 2 - Part 1 - for ticker",ticker)
    print(df_year45.head(5))
    print("\n")
    print("Question 2 - Part 2 - for ticker",ticker)  
    print(df_spy_year45.head(5))      
    print("\n")
    
    print("\n")
    print("Question 2 - Part 2 - for ticker",ticker,": percentage of true labels (both positive and negative) predicted correctly for the last two years for W=2 is",round(((w2_tm_accuracy_count)*100),2),"%")
    print("\n")
    print("Question 2 - Part 2 - for ticker",ticker,": percentage of true labels (both positive and negative) predicted correctly for the last two years for W=3 is ",round(((w3_tm_accuracy_count)*100),2),"%")
    print("\n")
    print("Question 2 - Part 2 - for ticker",ticker,": percentage of true labels (both positive and negative) predicted correctly for the last two years for W=4 is ",round(((w4_tm_accuracy_count)*100),2),"%")
    
    print("\n")
    print("Question 2 - Part 2 - for ticker",ticker2,": percentage of true labels (both positive and negative) predicted correctly for the last two years for W=2 is",round(((w2_spy_accuracy_count)*100),2),"%")
    print("\n")
    print("Question 2 - Part 2 - for ticker",ticker2,": percentage of true labels (both positive and negative) predicted correctly for the last two years for W=3 is ",round(((w3_spy_accuracy_count)*100),2),"%")
    print("\n")
    print("Question 2 - Part 2 - for ticker",ticker2,": percentage of true labels (both positive and negative) predicted correctly for the last two years for W=4 is ",round(((w3_spy_accuracy_count)*100),2),"%")
    """
    """
    
    print("\n")
    print("Question 2 - Part 3 - for ticker",ticker, "W=4 gave the best result wrt accuracy ")
    print("\n")
    print("Question 2 - Part 3 - for ticker",ticker2, "W=2 gave the best result wrt accuracy ")
    ##############################################################
    
    ##################### Question 3
    
    ##############################################################
    
    ## for TM ticker
    
    #  Definining a new columns for Ensemble method 
    df_year45['Ensemble']=''
    
    for i in range((len(df_year45))):
        s=df_year45.at[i,'W2']+df_year45.at[i,'W3']+df_year45.at[i,'W4']
        splus=s.count("+")
        sminus=s.count("-") 
        
        if splus > sminus:
            df_year45.at[i, 'Ensemble'] = '+'
        else:
            df_year45.at[i, 'Ensemble'] = '-'
    
    ## for SPY ticker
    
    #  Definining a new columns for Ensemble method 
    df_spy_year45['Ensemble']=''
    
    for i in range((len(df_spy_year45))):
        s=df_spy_year45.at[i,'W2']+df_spy_year45.at[i,'W3']+df_spy_year45.at[i,'W4']
        splus=s.count("+")
        sminus=s.count("-") 
        
        if splus > sminus:
            df_spy_year45.at[i, 'Ensemble'] = '+'
        else:
            df_spy_year45.at[i, 'Ensemble'] = '-'    
    
    ########### Question 3 Part 1
    
    print("\n")
    print("Question 3 - Part 1 - for ticker -",ticker,"on test data set")
    print(df_year45.head(5))
    print("\n")
    print("Question 3 - Part 1 - for ticker -",ticker,"on test data set")
    print(df_spy_year45.head(5))
    
    # for Percentage of True Vs Predicted label calculation 
    ##Initialize the count for 
    count_spy_ensemble=0    
    total_spy_count=len(df_spy_year45)
    for i in range((len(df_spy_year45))):
        if df_spy_year45.loc[i,"TrueLabel"]==df_spy_year45.loc[i,"Ensemble"]:
            count_spy_ensemble=count_spy_ensemble+1
        else:
            pass        

    # for Percentage of True Vs Predicted label calculation 
    ##Initialize the count for 
    count_ensemble=0
    
    total_count=len(df_year45)
    for i in range((len(df_year45))):
        # count True Lable Vs Predicted labe for W=2
        if df_year45.loc[i,"TrueLabel"]==df_year45.loc[i,"Ensemble"]:
            count_ensemble=count_ensemble+1
        else:
            pass
        
    ensemble_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["Ensemble"]=='+')]['Ensemble'].count()
    ensemble_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["Ensemble"]=='-')]['Ensemble'].count()
    ensemble_spy_accuracy_count= (ensemble_spy_tp_count + ensemble_spy_tn_count) /len(df_spy_year45)
    
    ensemble_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["Ensemble"]=='+')]['Ensemble'].count()
    ensemble_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["Ensemble"]=='-')]['Ensemble'].count()
    ensemble_tm_accuracy_count= (ensemble_tm_tp_count + ensemble_tm_tn_count) /len(df_year45)
    
    
    
    print("\n")
    print("Question 3 - Part 2 - for ticker",ticker,": percentage of true labels (both positive and negative) predicted correctly for the last two years by Ensenble method is",round(((ensemble_tm_accuracy_count)*100),2),"%")
        
    print("\n")
    print("Question 3 - Part 2 - for ticker",ticker2,": percentage of true labels (both positive and negative) predicted correctly for the last two years by Ensenble method is",round(((ensemble_spy_accuracy_count)*100),2),"%")
    print("\n")
    
    #### Question3 Part 3   
    
    
    #print("\n")
    print("Question 3 - Part 3 - for ticker",ticker,": accuracy of predicting [-] label using ensemble compared to W = 2,3,4 is decreased " )
        
    print("\n")
    print("Question 3 - Part 3 - for ticker",ticker2,": accuracy of predicting [-] label using ensemble compared to W = 3,4 has improved while wrt W=2 the accuracy is same as Ensamble")
    
        
    #### Question3 Part 4
    #print("\n")
    print("Question 3 - Part 4 - for ticker",ticker,": accuracy of predicting [+] label using ensemble compared to W = 2,3,4 is decreased " )
        
    print("\n")
    print("Question 3 - Part 4 - for ticker",ticker2,": accuracy of predicting [+] label using ensemble compared to W = 2,3,4 has not improved , W=2 giving better result wrt + labels vs ensemble")
    
    
    
    ######## Question 4
    
    ## Calcuation for SPY Ticker
    
    ### Question 4 - Part 1 - True Positive true positives (your predicted label is + and true labelis +
    w2_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W2"]=='+')]['W2'].count()
    w3_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W3"]=='+')]['W3'].count()
    w4_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W4"]=='+')]['W4'].count()
    ensemble_spy_tp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["Ensemble"]=='+')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 1 - TP for SPY Ticker W=2 is ", w2_spy_tp_count)
    print("Question 4 Part 1 - TP for SPY Ticker W=3 is ", w3_spy_tp_count)
    print("Question 4 Part 1 - TP for SPY Ticker W=4 is ", w4_spy_tp_count)
    print("Question 4 Part 1 - TP for SPY Ticker Ensemble is ", ensemble_spy_tp_count)
    
    
    ### Question 4 - Part 2 - FP - false positives (your predicted label is + but true labelis -
    w2_spy_fp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W2"]=='+')]['W2'].count()
    w3_spy_fp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W3"]=='+')]['W3'].count()
    w4_spy_fp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W4"]=='+')]['W4'].count()
    ensemble_spy_fp_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["Ensemble"]=='+')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 2 - FP for SPY Ticker W=2 is ", w2_spy_fp_count)
    print("Question 4 Part 2 - FP for SPY Ticker W=3 is ", w3_spy_fp_count)
    print("Question 4 Part 2 - FP for SPY Ticker W=4 is ", w4_spy_fp_count)
    print("Question 4 Part 2 - FP for SPY Ticker Ensemble is ", ensemble_spy_fp_count)
    
    
    ### Question 4 - Part 3 - TN - true negativess (your predicted label is - and truelabel is -
    w2_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W2"]=='-')]['W2'].count()
    w3_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W3"]=='-')]['W3'].count()
    w4_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["W4"]=='-')]['W4'].count()
    ensemble_spy_tn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='-') & (df_spy_year45["Ensemble"]=='-')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 3 - TN for SPY Ticker W=2 is ", w2_spy_tn_count)
    print("Question 4 Part 3 - TN for SPY Ticker W=3 is ", w3_spy_tn_count)
    print("Question 4 Part 3 - TN for SPY Ticker W=4 is ", w4_spy_tn_count)
    print("Question 4 Part 3 - TN for SPY Ticker Ensemble is ", ensemble_spy_tn_count)
    
    ### Question 4 - Part 4 - FN - false negatives (your predicted label is - but true label is +
    w2_spy_fn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W2"]=='-')]['W2'].count()
    w3_spy_fn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W3"]=='-')]['W3'].count()
    w4_spy_fn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["W4"]=='-')]['W4'].count()
    ensemble_spy_fn_count=df_spy_year45[(df_spy_year45["TrueLabel"]=='+') & (df_spy_year45["Ensemble"]=='-')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 4 - TN for SPY Ticker W=2 is ", w2_spy_fn_count)
    print("Question 4 Part 4 - TN for SPY Ticker W=3 is ", w3_spy_fn_count)
    print("Question 4 Part 4 - TN for SPY Ticker W=4 is ", w4_spy_fn_count)
    print("Question 4 Part 4 - TN for SPY Ticker Ensemble is ", ensemble_spy_fn_count)
    
    ### Question 4 - Part 5 - TPR = TP/(TP + FN) - true positive rate.
    w2_spy_tpr_count= w2_spy_tp_count/ ( w2_spy_tp_count+w2_spy_fn_count)
    w3_spy_tpr_count= w3_spy_tp_count/ (w3_spy_tp_count+w3_spy_fn_count) 
    w4_spy_tpr_count= w4_spy_tp_count/ (w4_spy_tp_count+w4_spy_fn_count)
    ensemble_spy_tpr_count= ensemble_spy_tp_count/ (ensemble_spy_tp_count+ensemble_spy_fn_count)
    
    print("\n")
    print("Question 4 Part 5 - TPR for SPY Ticker W=2 is ", round(w2_spy_tpr_count,4))
    print("Question 4 Part 5 - TPR for SPY Ticker W=3 is ", round(w3_spy_tpr_count,4))
    print("Question 4 Part 5 - TPR for SPY Ticker W=4 is ", round(w4_spy_tpr_count,4))
    print("Question 4 Part 5 - TPR for SPY Ticker Ensemble is ", round(ensemble_spy_tpr_count,4))
    
    ### Question 4 - Part 6 - TNR = TN/(TN + FP) - true negative rate.
    w2_spy_tnr_count= w2_spy_tn_count/ ( w2_spy_tn_count+w2_spy_fp_count)
    w3_spy_tnr_count= w3_spy_tn_count/ ( w3_spy_tn_count+w3_spy_fp_count) 
    w4_spy_tnr_count= w4_spy_tn_count/ ( w4_spy_tn_count+w4_spy_fp_count)
    ensemble_spy_tnr_count= ensemble_spy_tn_count/ ( ensemble_spy_tn_count+ensemble_spy_fp_count)
    
    print("\n")
    print("Question 4 Part 6 - TNR for SPY Ticker W=2 is ", round(w2_spy_tnr_count,4))
    print("Question 4 Part 6 - TNR for SPY Ticker W=3 is ", round(w3_spy_tnr_count,4))
    print("Question 4 Part 6 - TNR for SPY Ticker W=4 is ", round(w4_spy_tnr_count,4))
    print("Question 4 Part 6 - TNR for SPY Ticker Ensemble is ", round(ensemble_spy_tnr_count,4))
    
    ### Question 4 - Calclating Accuracy = (TP + TN)/All
    w2_spy_accuracy_count= ( w2_spy_tp_count + w2_spy_tn_count ) / len(df_spy_year45)
    w3_spy_accuracy_count= ( w3_spy_tp_count + w3_spy_tn_count ) / len(df_spy_year45)
    w4_spy_accuracy_count= ( w4_spy_tp_count + w4_spy_tn_count ) / len(df_spy_year45)
    ensemble_spy_accuracy_count= (ensemble_spy_tp_count + ensemble_spy_tn_count) /len(df_spy_year45)
    
    ########## for TM Ticker
    
    ### Question 4 - Part 1 - True Positive true positives (your predicted label is + and true labelis +
    w2_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W2"]=='+')]['W2'].count()
    w3_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W3"]=='+')]['W3'].count()
    w4_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W4"]=='+')]['W4'].count()
    ensemble_tm_tp_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["Ensemble"]=='+')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 1 - TP for TM Ticker W=2 is ", w2_tm_tp_count)
    print("Question 4 Part 1 - TP for TM Ticker W=3 is ", w3_tm_tp_count)
    print("Question 4 Part 1 - TP for TM Ticker W=4 is ", w4_tm_tp_count)
    print("Question 4 Part 1 - TP for TM Ticker Ensemble is ", ensemble_tm_tp_count)
    
    ### Question 4 - Part 2 - FP - false positives (your predicted label is + but true labelis -
    w2_tm_fp_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W2"]=='+')]['W2'].count()
    w3_tm_fp_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W3"]=='+')]['W3'].count()
    w4_tm_fp_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W4"]=='+')]['W4'].count()
    ensemble_tm_fp_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["Ensemble"]=='+')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 2 - FP for TM Ticker W=2 is ", w2_tm_fp_count)
    print("Question 4 Part 2 - FP for TM Ticker W=3 is ", w3_tm_fp_count)
    print("Question 4 Part 2 - FP for TM Ticker W=4 is ", w4_tm_fp_count)
    print("Question 4 Part 2 - FP for TM Ticker Ensemble is ", ensemble_tm_fp_count)
    
    
    ### Question 4 - Part 3 - TN - true negativess (your predicted label is - and truelabel is -
    w2_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W2"]=='-')]['W2'].count()
    w3_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W3"]=='-')]['W3'].count()
    w4_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["W4"]=='-')]['W4'].count()
    ensemble_tm_tn_count=df_year45[(df_year45["TrueLabel"]=='-') & (df_year45["Ensemble"]=='-')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 3 - TM for TM Ticker W=2 is ", w2_tm_tn_count)
    print("Question 4 Part 3 - TN for TM Ticker W=3 is ", w3_tm_tn_count)
    print("Question 4 Part 3 - TN for TM Ticker W=4 is ", w4_tm_tn_count)
    print("Question 4 Part 3 - TN for TM Ticker Ensemble is ", ensemble_tm_tn_count)
    
    ### Question 4 - Part 4 - FN - false negatives (your predicted label is - but true label is +
    w2_tm_fn_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W2"]=='-')]['W2'].count()
    w3_tm_fn_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W3"]=='-')]['W3'].count()
    w4_tm_fn_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["W4"]=='-')]['W4'].count()
    ensemble_tm_fn_count=df_year45[(df_year45["TrueLabel"]=='+') & (df_year45["Ensemble"]=='-')]['Ensemble'].count()
    
    print("\n")
    print("Question 4 Part 4 - FM for TM Ticker W=2 is ", w2_tm_fn_count)
    print("Question 4 Part 4 - FN for TM Ticker W=3 is ", w3_tm_fn_count)
    print("Question 4 Part 4 - FN for TM Ticker W=4 is ", w4_tm_fn_count)
    print("Question 4 Part 4 - FN for TM Ticker Ensemble is ", ensemble_tm_fn_count)
    
    
    ### Question 4 - Part 5 - TPR = TP/(TP + FN) - true positive rate.
    w2_tm_tpr_count= w2_tm_tp_count/ ( w2_tm_tp_count+w2_tm_fn_count)
    w3_tm_tpr_count= w3_tm_tp_count/ (w3_tm_tp_count+w3_tm_fn_count) 
    w4_tm_tpr_count= w4_tm_tp_count/ (w4_tm_tp_count+w4_tm_fn_count)
    ensemble_tm_tpr_count= ensemble_tm_tp_count/ (ensemble_tm_tp_count+ensemble_tm_fn_count)
    
    print("\n")
    print("Question 4 Part 5 - TPR for TM Ticker W=2 is ", round(w2_tm_tpr_count,3))
    print("Question 4 Part 5 - TPR for TM Ticker W=3 is ", round(w3_tm_tpr_count,3))
    print("Question 4 Part 5 - TPR for TM Ticker W=4 is ", round(w4_tm_tpr_count,3))
    print("Question 4 Part 5 - TPR for TM Ticker Ensemble is ", round(ensemble_tm_tpr_count,3))
    
    ### Question 4 - Part 6 - TNR = TN/(TN + FP) - true negative rate.
    w2_tm_tnr_count= w2_tm_tn_count/ ( w2_tm_tn_count+w2_tm_fp_count)
    w3_tm_tnr_count= w3_tm_tn_count/ ( w3_tm_tn_count+w3_tm_fp_count) 
    w4_tm_tnr_count= w4_tm_tn_count/ ( w4_tm_tn_count+w4_tm_fp_count)
    ensemble_tm_tnr_count= ensemble_tm_tn_count/ ( ensemble_tm_tn_count+ensemble_tm_fp_count)
    
    print("\n")
    print("Question 4 Part 6 - TNR for TM Ticker W=2 is ", round(w2_tm_tnr_count,3))
    print("Question 4 Part 6 - TNR for TM Ticker W=3 is ", round(w3_tm_tnr_count,3))
    print("Question 4 Part 6 - TNR for TM Ticker W=4 is ", round(w4_tm_tnr_count,3))
    print("Question 4 Part 6 - TNR for TM Ticker Ensemble is ", round(ensemble_tm_tnr_count,3))
    
    ### Question 4 - Calclating Accuracy = (TP + TN)/All
    w2_tm_accuracy_count= ( w2_tm_tp_count + w2_tm_tn_count ) / len(df_year45)
    w3_tm_accuracy_count= ( w3_tm_tp_count + w3_tm_tn_count ) / len(df_year45)
    w4_tm_accuracy_count= ( w4_tm_tp_count + w4_tm_tn_count ) / len(df_year45)
    ensemble_tm_accuracy_count= (ensemble_tm_tp_count + ensemble_tm_tn_count) /len(df_year45)
    
    ### Question 4 - Part 4 - Summary
     
    dfn = pd.DataFrame({'W' : ['2', '3', '4','Ensemble','2', '3', '4','Ensemble'],
                        'Ticker' : ['S&P-500', 'S&P-500', 'S&P-500','S&P-500','TM', 'TM', 'TM','TM'],
                        'TP':[w2_spy_tp_count, w3_spy_tp_count, w4_spy_tp_count,ensemble_spy_tp_count,
                              w2_tm_tp_count,w3_tm_tp_count,w4_tm_tp_count,ensemble_tm_tp_count],
                       'FP':[w2_spy_fp_count, w3_spy_fp_count, w4_spy_fp_count,ensemble_spy_fp_count,
                              w2_tm_fp_count,w3_tm_fp_count,w4_tm_fp_count,ensemble_tm_fp_count],
                       'TN':[w2_spy_tn_count, w3_spy_fp_count, w4_spy_tn_count,ensemble_spy_tn_count,
                              w2_tm_tn_count,w3_tm_tn_count,w4_tm_tn_count,ensemble_tm_tn_count],
                       'FN':[w2_spy_fn_count, w3_spy_fp_count, w4_spy_fn_count,ensemble_spy_fn_count,
                              w2_tm_fn_count,w3_tm_fn_count,w4_tm_fn_count,ensemble_tm_fn_count],
                       'Accuracy':[w2_spy_accuracy_count, w3_spy_accuracy_count, w4_spy_accuracy_count,ensemble_spy_accuracy_count,
                              w2_tm_accuracy_count,w3_tm_accuracy_count,w4_tm_accuracy_count,ensemble_tm_accuracy_count],
                       'TPR':[w2_spy_tpr_count, w3_spy_tpr_count, w4_spy_tpr_count,ensemble_spy_tpr_count,
                              w2_tm_tpr_count,w3_tm_tpr_count,w4_tm_tpr_count,ensemble_tm_tpr_count],
                       'TNR':[w2_spy_tnr_count, w3_spy_tnr_count, w4_spy_tnr_count,ensemble_spy_tnr_count,
                              w2_tm_tnr_count,w3_tm_tnr_count,w4_tm_tnr_count,ensemble_tm_tnr_count],
                       })
    
    print("\n")
    
    print("Question 4 - Part 7")
    
    ## Printing the summary output - Question 4 - Part 4 - Summary
    print(dfn) 
    
    
    print("\n")
    
    print("Question 4 - Part 8 - for ticker",ticker,"The obeservation is that accuracy is lower than S&P Ticker. Also W=4 is giving relative good accuracy with respect to other methods, overall the prediction is not good TPR is also very low")
    
    print("\n")
    
    print("Question 4 - Part 8 - for ticker",ticker2,"The observation is that accuracy is lower than S&P Ticker. Also W=4 is giving relative good accuracy with respect to other methods, overall the prediction is not good ,TPR is also very low.")
    
        
    ## Saving the TM Sticker dataframe for use in question 6
    object_df_year45 = df_year45
    file_tm_year45 = open('object_df_year45.obj', 'wb') 
    # save TM
    pickle.dump(object_df_year45, file_tm_year45)
    file_tm_year45.close()

    ## Saving the SPY Sticker dataframe  for use in quesiton 6  
    object_df_spy_year45 = df_spy_year45
    file_spy_year45 = open('object_df_spy_year45.obj', 'wb') 
    # save TM
    pickle.dump(object_df_spy_year45, file_spy_year45)
    file_spy_year45.close()
    

    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)