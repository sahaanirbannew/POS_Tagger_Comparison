def train_configuration(num):

    '''
    Train configuration is used to train the Penn treebank, Genia and CoNLL dataset with the given Data Representation Format (DRF).
    
    input : num 
                The configuration number of the DRF
                
    return : None
    
    '''
    start_time = time.time()
    #Read the DRF files
    
    drf_path = input("Enter the DRF folder path for the required configuration : ")
    drf_conf_path = drf_path + "{}".format(num)
    drf_files = get_drf(drf_conf_path)
    
    
    # Reading the DRF dataset

    drf_penn_train = drf_files['drf_penn_train.pkl']
    drf_penn_test = drf_files['drf_penn_test.pkl']

    drf_genia_train = drf_files['drf_genia_train.pkl']
    drf_genia_test = drf_files['drf_genia_test.pkl']

    drf_conll_train = drf_files['drf_conll_train.pkl']
    drf_conll_test = drf_files['drf_conll_test.pkl']
    
    #Print the length of the Training and Testing datasets

    print("Lines in Penn treebank training dataset"+ str(len(penn_train_sequence)))
    print("Lines in Penn treebank testing dataset" + str(len(penn_test_sequence)))
    print("Lines in Genia training dataset "+ str(len(genia_train_sequence)))
    print("Lines in Genia testing dataset "+ str(len(genia_test_sequence)))
    print("Lines in CoNLL training dataset "+ str(len(conll_train_sequence)))
    print("Lines in CoNLL testing dataset "+ str(len(conll_test_sequence)))

    # Creating a training and testing dataset

    x_penn_train, y_penn_train = transform_to_dataset(penn_train_sequence)
    x_penn_test, y_penn_test = transform_to_dataset(penn_test_sequence)

    x_genia_train, y_genia_train = transform_to_dataset(genia_train_sequence)
    x_genia_test, y_genia_test = transform_to_dataset(genia_test_sequence)

    x_conll_train, y_conll_train = transform_to_dataset(conll_train_sequence)
    x_conll_test, y_conll_test = transform_to_dataset(conll_test_sequence)

    # Penn treebank CRF model
    ptb_crf_model = penn_tree_model(x_penn_train,y_penn_train,x_penn_test,y_penn_test)

    # Genia CRF model
    genia_crf_model = genia_model(x_genia_train,y_genia_train,x_genia_test,y_genia_test)
    
    #CoNLL CRF model 
    conll_crf_model = conll_model(x_conll_train,y_conll_train,x_conll_test,y_conll_test)
    
    #Penn treebank CRF model on Genia dataset 
    ptb_genia_model(x_genia_test,y_genia_test,ptb_crf_model)
    
    #Penn treebank CRF model on CoNLL dataset 
    ptb_conll_model(x_conll_test,y_conll_test,ptb_crf_model)
    
    #Genia CRF model on Penn treebank dataset 
    genia_penn_model(x_penn_test,y_penn_test,genia_crf_model)
    
    #Genia CRF model on CoNLL dataset
    genia_conll_model(x_conll_test,y_conll_test,genia_crf_model)
    
    # CoNLL CRF model on Penn Treebank dataset
    conll_ptb_model(x_penn_test,y_penn_test,conll_crf_model)

    # CoNLL CRF model on Genia dataset
    conll_genia_model(x_genia_test,y_genia_test,conll_crf_model)
    
    with open('cross_model_eval_config_{}.txt'.format(num), 'w') as f:
         print("Penn Treebank on Genia", file=f)
         print(ptb_genia['flat_accuracy'], file=f)
         print(ptb_genia['sequence_accuracy'], file=f)
         print(ptb_genia['report'], file=f)

         print("Penn Treebank on CoNLL", file=f)
         print(ptb_conll['flat_accuracy'], file=f)
         print(ptb_conll['sequence_accuracy'], file=f)
         print(ptb_conll['report'], file=f)
        
         print("Genia on Penn Treebank", file=f)
         print(genia_penn['flat_accuracy'], file=f)
         print(genia_penn['sequence_accuracy'], file=f)
         print(genia_penn['report'], file=f)

         print("Genia on CoNLL ", file=f)
         print(genia_conll['flat_accuracy'], file=f)
         print(genia_conll['sequence_accuracy'], file=f)
         print(genia_conll['report'], file=f)

         print("CoNLL on Penn Treebank ", file=f)
         print(conll_ptb['flat_accuracy'], file=f)
         print(conll_ptb['sequence_accuracy'], file=f)
         print(conll_ptb['report'], file=f)

        print("CoNLL on  Genia ", file=f)
        print(conll_genia['flat_accuracy'], file=f)
        print(conll_genia['sequence_accuracy'], file=f)
        print(conll_genia['report'], file=f)
        
    end_time = time.time()
    
    total_time = end_time - start_time 

    print(" Total time for this conf : " +str(total_time))

