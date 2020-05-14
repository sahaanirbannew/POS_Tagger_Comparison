#This file defines the functions to train CRF models on different dataset 

#Penn treebank CRF model
def penn_tree_model(x_penn_train,y_penn_train,x_penn_test,y_penn_test):
    '''
    Penn Treebank CRF model 
    
    input : Training dataset 
                x_penn_train  = training sequence 
                y_penn_train  = labels of the training sequence 
                x_penn_test   = testing sequence 
                y_penn_test   = labels of the testing sequence  
                
    output : Penn treebank CRF  Model 
                 ptb_crf_model
    '''
    

    #Grid search for C2 value

    crf_model = CRF(algorithm='l2sgd', all_possible_states=True, all_possible_transitions= True,verbose= False, period = 30)

    params_space = {  'c2': [0.001,0.01,0.1,0.05,0.0005]   }    # how to choose c2 value?
                               
    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_accuracy_score)

    # Grid search
    rs = GridSearchCV(crf_model,params_space,scoring=f1_scorer, n_jobs= -1, cv=2, verbose=1,return_train_score=True)

    # intermediate model for C2 value selection 
    rs.fit(x_penn_train, y_penn_train)

    # using the best C2 value to train the model 
    ptb_crf_model = CRF( algorithm='l2sgd' ,c2 = rs.best_estimator_.c2 , all_possible_states=True, all_possible_transitions= True,verbose= True, period = 30)
    ptb_crf_model.fit(x_penn_train, y_penn_train, x_penn_test , y_penn_test)
    
    #testing the model 
    y_penn_pred = ptb_crf_model.predict(x_penn_test)
    
    #Print the Results 
    print("Accuracy  :  " + str(metrics.flat_accuracy_score(y_penn_test, y_penn_pred)))
    print("Sequence Accuracy :  " + str(metrics.sequence_accuracy_score(y_penn_test, y_penn_pred)))
    print("Classification Report")
    print(metrics.flat_classification_report(y_penn_test, y_penn_pred))
    
    print("The trained model for the given configuration is stored in the current directory")
    f = open('ptb_crf_model_config_{}.pkl'.format(num),'wb')
    pickle.dump(ptb_crf_model,f)

    plot_model_evaluation(ptb_crf_model.training_log_.iterations,num,'Penn Treebank')

    return ptb_crf_model

 #Genia treebank CRF model
def genia_model(x_genia_train,y_genia_train,x_genia_test,y_genia_test):

    '''
   Genia CRF model 
    
    input : Training dataset 
                x_genia_train  = training sequence 
                y_genia_train  = labels of the training sequence 
                x_genia_test   = testing sequence 
                y_genia_test   = labels of the testing sequence  
                
    output : Genia CRF  Model 
                 genia_crf_model
    '''

    #Grid search for C2 value
    crf_model = CRF(algorithm='l2sgd', all_possible_states=True, all_possible_transitions= True,verbose= False, period = 30)
    params_space = {'c2': [0.001,0.01,0.1,0.05,0.0005] }     # how to choose c2 value?

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_accuracy_score)

    # Grid search
    rs = GridSearchCV(crf_model,params_space,scoring=f1_scorer, n_jobs= -1, cv=2, verbose=1,return_train_score=True)
    rs.fit(x_genia_train, y_genia_train)
    
    # using the best C2 value to train the model 
    genia_crf_model = CRF(algorithm='l2sgd',c2= rs.best_estimator_.c2 , all_possible_states=True, all_possible_transitions= True,verbose= True, period = 30)
    genia_crf_model.fit(x_genia_train, y_genia_train, x_genia_test, y_genia_test)

    # testing the trained model
    y_genia_pred = genia_crf_model.predict(x_genia_test)
    
    print("Accuracy : " + str(metrics.flat_accuracy_score(y_genia_test, y_genia_pred)))
    print("Sequence Accuracy : " + str(metrics.sequence_accuracy_score(y_genia_test, y_genia_pred)))
    print("Classification report")
    print(metrics.flat_classification_report(y_genia_test, y_genia_pred))

    print("The trained model for the given configuration is stored in the current directory")
    f = open('genia_crf_model_config_{}.pkl'.format(num),'wb')
    pickle.dump(genia_crf_model,f)

    plot_model_evaluation(genia_crf_model.training_log_.iterations,num,'Genia')

    return genia_crf_model

 #Conll treebank CRF model
def conll_model(x_conll_train,y_conll_train,x_conll_test,y_conll_test):
 '''
   CoNLL CRF model 
    
    input : Training dataset 
                x_genia_train  = training sequence 
                y_genia_train  = labels of the training sequence 
                x_genia_test   = testing sequence 
                y_genia_test   = labels of the testing sequence  
                
    output : CoNLL CRF  Model 
                 conll_crf_model
    '''

    #Grid search for C2 value
    crf_model = CRF(algorithm='l2sgd', all_possible_states=True, all_possible_transitions= True,verbose= False, period = 30)
    params_space = {'c2': [0.001,0.01,0.1,0.05,0.0005] }     # how to choose c2 value?

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_accuracy_score)

    # Grid search
    rs = GridSearchCV(crf_model,params_space,scoring=f1_scorer, n_jobs= -1, cv=2, verbose=1,return_train_score=True)
    rs.fit(x_conll_train, y_conll_train)

    # using the best C2 value to train the model 
    conll_crf_model = CRF(algorithm='l2sgd',c2=rs.best_estimator_.c2, all_possible_states= True, all_possible_transitions= True,verbose= True, period = 30)
    conll_crf_model.fit(x_conll_train, y_conll_train, x_conll_test, y_conll_test)

    # testing the training model
    y_conll_pred = conll_crf_model.predict(x_conll_test)
    
    print("Accuracy : " + str(metrics.flat_accuracy_score(y_conll_test, y_conll_pred)))
    print("Sequence Accuracy : " + str(metrics.sequence_accuracy_score(y_conll_test, y_conll_pred)))
    print("Classification report")
    print(metrics.flat_classification_report(y_conll_test, y_conll_pred))

    print("The trained model for the given configuration is stored in the current directory")
    f = open('conll_crf_model_config_{}.pkl'.format(num),'wb')
    pickle.dump(conll_crf_model,f)

    plot_model_evaluation(conll_crf_model.training_log_.iterations,num,'CoNLL')

    return conll_crf_model


#Cross domain testing of the models 

def ptb_genia_model(x_genia_test,y_genia_test,ptb_crf_model):

    # ptb crf model on Genia dataset

    ptb_genia_pred = ptb_crf_model.predict(x_genia_test)


    print("Accuracy : " + str(metrics.flat_accuracy_score(y_genia_test, ptb_genia_pred)))
    print("Sequence Accuracy : " + str(metrics.sequence_accuracy_score(y_genia_test, ptb_genia_pred)))
    print("Classification Report")
    print(metrics.flat_classification_report(y_genia_test, ptb_genia_pred))


def ptb_conll_model(x_conll_test,y_conll_test,ptb_crf_model):

    # ptb crf model on CoNLL dataset
    
    ptb_conll_pred = ptb_crf_model.predict(x_conll_test)


    print("Accuracy : " + str(metrics.flat_accuracy_score(y_conll_test, ptb_conll_pred)))
    print("Sequence Accuracy : " + str(metrics.sequence_accuracy_score(y_conll_test, ptb_conll_pred)))
    print("Classifcation report")
    print(metrics.flat_classification_report(y_conll_test, ptb_conll_pred))


def genia_penn_model(x_penn_test,y_penn_test,genia_crf_model):
    # genia crf model on Penn tree bank dataset

    genia_ptb_pred = genia_crf_model.predict(x_penn_test)


    print("Accuracy : " +str(metrics.flat_accuracy_score(y_penn_test, genia_ptb_pred)))
    print("Sequence Accuracy : " +str(metrics.sequence_accuracy_score(y_penn_test, genia_ptb_pred)))
    print("Classification Report")
    print(metrics.flat_classification_report(y_penn_test, genia_ptb_pred))

def genia_conll_model(x_conll_test,y_conll_test,genia_crf_model):
    # genia crf model on CoNLL dataset
    genia_conll_pred = genia_crf_model.predict(x_conll_test)


    print("Accuracy : " +str(metrics.flat_accuracy_score(y_conll_test, genia_conll_pred)))
    print("Sequence Accuracy : " + str(metrics.sequence_accuracy_score(y_conll_test, genia_conll_pred)))
    print("Classification report" )
    print(metrics.flat_classification_report(y_conll_test, genia_conll_pred))

def conll_ptb_model(x_penn_test,y_penn_test,conll_crf_model):
    # CoNLL crf model on Penn tree bank dataset

    conll_ptb_pred = conll_crf_model.predict(x_penn_test)


    print("Accuracy : " + str(metrics.flat_accuracy_score(y_penn_test, conll_ptb_pred)))
    print("Sequence Accuracy : " + str(metrics.sequence_accuracy_score(y_penn_test, conll_ptb_pred)))
    print("Classification report")
    print(metrics.flat_classification_report(y_penn_test, conll_ptb_pred))


def conll_genia_model(x_genia_test,y_genia_test,conll_crf_model):
    # CoNLL crf model on Genia dataset

    conll_genia_pred = conll_crf_model.predict(x_genia_test)


    print("Accuracy : " + str(metrics.flat_accuracy_score(y_genia_test, conll_genia_pred)))
    print("Sequence Accuracy : " + str(metrics.sequence_accuracy_score(y_genia_test, conll_genia_pred)))
    print("Classification report")
    print(metrics.flat_classification_report(y_genia_test, conll_genia_pred))
