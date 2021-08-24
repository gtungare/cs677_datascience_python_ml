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
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split

url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00519//heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(url,delimiter=',')
df = pd.DataFrame(data)

np.random.seed(10001)

try:   
    
    ### Question  1 part 1
    
       
    print("Question  1 part 1")
    print("\n")
    
    # Extract two dataframes with the above 4 features: df 0 for surviving patients (DEATH EVENT = 0) and df 1 for deceased patients (DEATH EVENT = 1)
    df_0=df[["creatinine_phosphokinase","serum_creatinine","serum_sodium","platelets"]][df.DEATH_EVENT ==0]
    df_1=df[["creatinine_phosphokinase","serum_creatinine","serum_sodium","platelets"]][df.DEATH_EVENT ==1]
    
    print("Question  1 part 1 - print df_0")
    print(df_0.head(5))
    print("\n")
    
    print("Question  1 part 1 - print df_1")
    print(df_1.head(5))
    print("\n")
    
    print("Question  1 part 2")
    print("\n")
    # correlation matrices
    fig_pairplot_df0 = sns.pairplot(df_0)
    plt.show()
    fig_pairplot_df0.savefig("Corr_df_0.pdf", bbox_inches='tight')
    
    fig_pairplot_df1 = sns.pairplot(df_1)
    plt.show()
    fig_pairplot_df1.savefig("Corr_df_1.pdf", bbox_inches='tight')
    
    # Compute the correlation matrix
    corr = df_0.corr()
    
    print("correlation matrix : df_0")
    print(corr)
    print("\n")
    corr_heatmap_df_0 = sns.heatmap(df_0.corr())
    plt.show()
    #corr_heatmap_df_0.savefig("Corr_heatmap_df_0.pdf", bbox_inches='tight')
    
    # Compute the correlation matrix
    corr_df1 = df_1.corr()
    
    print("correlation matrix : df_1")
    print(corr_df1)
    print("\n")
    corr_heatmap_df_1 = sns.heatmap(df_1.corr())
    plt.show()
    #corr_heatmap_df_1.savefig("Corr_heatmap_df_1.pdf", bbox_inches='tight')
    

    
    print("Question  1 part 3")
    print("\n")
    
    print("Question  1 part 3 (a) which features have the highest correlation for survivingpatients?")
    print("Comments - serum_creatine & creatinine_phosphokinase have the higest correlation for surviving patients") 
    print("\n")    
    
    print("Question  1 part 3 (b) which features have the lowest correlation for surviving patients?")
    print("Comments - creatinine_phosphokinase and platelets have the lowest correlation for surviving patients, hovering around 0.001") 
    print("\n")    
    
    print("Question  1 part 3 (c) which features have the highest correlation for deceased patients?")
    print("Comments - creatinine_phosphokinase and serum_sodium have the highest correlation among the deceased patients ") 
    print("\n")    
    
    print("Question  1 part 3 (d) which features have the lowest correlation for deceased patients?")
    print("Comments - serum_creatinine & platelets have the lowest correlation for deceased patients ") 
    print("\n")    
    
    print("Question  1 part 3 (e) are results the same for both cases?")
    print("Comments - No the results are different for both cases ") 
    print("\n")    

    
    #######################################
    
    print("Question  2")
    print("\n")
    
    """
    Function is written to calcaute SSE and return SSE and weights
    THe function accepets DEATH_EVENT = 0 or 1 as input 
    the function has another input deg which is custom variable which deifnes the degree of function/model/polynomial 
    
    """
    
    ##  Defining function to  calculate SSE for Surviing and deceaased patient for different model
    
    def myfunct(event,deg):
        ### Group 2: X: platelets, Y : serum sodium
        X = df["platelets"][df.DEATH_EVENT == event].values
        Y = df["serum_sodium"][df.DEATH_EVENT ==event].values
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.50)
            
        degree = deg
        if degree == 1 or degree == 2 or degree == 3:
            weights = np.polyfit(X_train,Y_train, degree)
        elif degree == 4:
            weights = np.polyfit(np.log(X_train),Y_train, 1)
        else:
            weights = np.polyfit(np.log(X_train),np.log(Y_train), 1)
            
        model = np.poly1d(weights)
        predicted = model(X_test)        
    
        plt.plot(predicted,X_test,label='Predicted Vs X_test')
        plt.xlabel("predicted")
        plt.ylabel("X_test")
        ## Caclualate Loss
        lse=sum((Y_test-predicted)*(Y_test-predicted))

        
        return (lse,weights)
    
    # Calling func for y=ax+b death_event=0
    slr_0=myfunct(event=0,deg=1)
    # Calling func for y=ax+b death_event=1
    slr_1=myfunct(event=1,deg=1)
    # Calling func for y = ax2 + bx + c death_event=0
    quadratic_0=myfunct(event=0,deg=2)
    # Calling func for y = ax2 + bx + c death_event=1
    quadratic_1=myfunct(event=1,deg=2)
    # Calling func for y = ax3 + bx2 + cx + d death_event=0
    cubic_0=myfunct(event=0,deg=3)
    # Calling func for y = ax3 + bx2 + cx + d death_event=1
    cubic_1=myfunct(event=1,deg=3)
    # Calling func for y = a log x + b death_event=0
    log_0=myfunct(event=0,deg=4)
    # Calling func for y = a log x + b death_event=1
    log_1=myfunct(event=1,deg=4)
    # Calling func for log y = a log x + b death_event=0
    glm_0=myfunct(event=0,deg=5)
    # Calling func for log y = a log x + b death_event=1
    glm_1=myfunct(event=1,deg=5)
    
    ###
    
    ## SSE and Weights for Surv and Deceased are captured in one dataframe
    
    dfn = pd.DataFrame({'Model' : ['y = ax + b', 'y = ax2 + bx + c',
                                   'y = ax3 + bx2 + cx + d','y = a log x + b','log y = a log x + b'],
                        'SSE (death event=0)':[round(slr_0[0],3),round(quadratic_0[0],3),round(cubic_0[0],3),round(log_0[0],3),round(glm_0[0],3)],
                        'SSE (death event=1)':[round(slr_1[0],3),round(quadratic_1[0],3),round(cubic_1[0],3),round(log_1[0],3),round(glm_1[0],3)],
                        'Weights (death event=0)':[slr_0[1],quadratic_0[1],cubic_0[1],log_0[1],glm_0[1]],
                        'Weights (death event=1)':[slr_1[1],quadratic_1[1],cubic_1[1],log_1[1],glm_1[1]]
                        
                       })
    print("\n")
    print("Question 2 -(b) - printing the weights")
    print("\n")
    print(dfn[["Model","Weights (death event=0)"]])
    print("\n")
    print(dfn[["Model","Weights (death event=1)"]])
    print("\n")
    print("\n")
    print("Question 3 -Summary result ")
    print("\n")
    print(dfn[["Model","SSE (death event=0)","SSE (death event=1)"]])
    print("\n")
    print("Question 3 -part 1 - which model was the best (smallest SSE) for surviving patients? for deceased patients?  ")
    print("Comments - The model which performs best is the Simple Linear regression y =ax+B with smallest SSE")
    print("\n")
    print("Question 3 -part2 - which model was the worst (largest SSE) for surving patients? for deceased patients? ")
    print("Comments - The model which performs worst is y = a log x + b with largest SSE")
    print("\n")
    
except Exception as e:
    print(e)
    print('failed to excute correctly')