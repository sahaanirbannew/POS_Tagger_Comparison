'''
Developer         :      Venkatesh Murugadas
Version           :      0.03
Date              :      05/04/2020
Input             :      Corpus file path of Penn Treebank, Genia and CoNLL
Output            :      Dataset files of Penn Treebank, Genia and CoNLL in pickle format
Description       :      This file calls the functions from the corpus_reader.py for parsing the pentree bank, Genia and CoNLL corpus used for Parts of Speech Tagging.
Documentation Link:      https://docs.google.com/document/d/16GPkM6cVHHhpFTJk_WOIH4i23Ap9GiHmSvzJfmZYHRE/edit?usp=sharing

Version History   :
- 0.02
- 02/03/2020
- Adding the function for test train split of the dataset
- Change ID : #TestTrainSplit

#TODO

    1. Incorporate Log function from global.py

Main Program

Pre-requistie for running the program
    1. The file path for the Penn Treebank, Genia and CoNLL corpus for training.

'''
from corpus_reader import *

option = ''

while option.lower() != 'exit':

    # Modified tag dictionary is written in various files for the convience of the user.
    data_file_path = "./dataset/"

    # check whether the dataset folder is created or not
    if not os.path.exists(os.path.dirname(data_file_path)):
        try:
            os.makedirs(os.path.dirname(data_file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    option = read_corpus()

    if option == '1':

        """

        PENN TREE BANK CORPUS

        """

        penn_tree_file_path = input("Please Enter the file Path for Penn Tree bank : ")
        ptb_line_number =  int(input("Please Enter the Line number to check the corpus : "))

        print(" Reading the Penn Treebank dataset ")
        # Read the files from the folder
        ptb_files = get_data(penn_tree_file_path)
        # Create the initial dictionary from the corpus
        ptb_sentence_dict = create_penntreebank_dataset(ptb_files)
        # Statistical data of  word list and word counter
        ptb_word_list, ptb_word_counter = create_word_list(ptb_sentence_dict)
        # Statistical data of  tag list and tag counter
        ptb_tag_list, ptb_tag_counter = create_tag_list(ptb_sentence_dict)
        # Statistical data of sentence list
        ptb_sentences_list = create_sentences_list(ptb_sentence_dict)

        print("********************************************************")
        print("Initial Dictionary created")
        print("Number of Sentences : {}" .format(len(ptb_sentences_list)))
        print("Number of Words : {}" .format(len(ptb_word_list)))
        print("Number of Initial Tags : {}".format(len(ptb_tag_counter)))
        print("Initial Tag list ")

        pprint.pprint(ptb_tag_counter)

        print("Line : {} " .format(len(ptb_sentence_dict)))

        print("Initial format of the corpus with umodified tags")
        # Insert the line_number to view the line in the dataset
        print(ptb_sentence_dict[ptb_line_number])
        print("********************************************************")

        #intermediate storage point of reading the corpus. The corpus is written to a pickle file and then from here it is continued for further process
        ptb_file_name = 'dataset/penntreebank.pkl'
        # write the dictionary to pickle file
        write_pickle(ptb_file_name,ptb_sentence_dict)
        # the dictionary file read from the pickle file
        ptb_data_dictionary = read_pickle(ptb_file_name)
        ptb_words_list , ptb_words_counter = create_word_list(ptb_data_dictionary)
        ptb_tags_list, ptb_tags_counter = create_tag_list(ptb_data_dictionary)
        ptb_sentences_list_new = create_sentences_list(ptb_data_dictionary)

        print("********************************************************")
        print("Intermediate Pickle file created with unmodified tags")
        print("********************************************************")

        #Create a modified tag dictionary from the tagset provided in the corpus to Universal POS tags

        ptb_modified_dictionary = modify_tags(ptb_data_dictionary)
        ptb_modified_words_list , ptb_modified_words_counter = create_word_list(ptb_modified_dictionary)
        ptb_modified_tags_list , ptb_modified_tags_counter = create_tag_list(ptb_modified_dictionary)
        ptb_modified_sentences_list = create_sentences_list(ptb_data_dictionary)
        penn_train_dictionary , penn_test_dictionary = test_train_split(ptb_modified_dictionary)

        print("********************************************************")
        print("Modified Dictionary Created")
        print("Number of Initial Tags : {}".format(len(ptb_modified_tags_counter)))
        print("Modified Tag list")
        pprint.pprint(ptb_modified_tags_counter)

        print("Final Format of the corpus with Modified tags")
         # Insert the line_number to view the line in the dataset
        print(penn_train_dictionary[ptb_line_number])
        # print(ptb_modified_dictionary[ptb_line_number])
        print("********************************************************")

        # Write output files

        # write_txt(data_file_path+'dataset_penntreebank.txt', ptb_modified_sentences_list)
        # write_csv(data_file_path+'dataset_penntreebank',ptb_data_dictionary)
        # write_tsv(data_file_path+'dataset_penntreebank',ptb_data_dictionary)
        write_pickle(data_file_path+'dataset_penn_train.pkl',penn_train_dictionary)
        write_pickle(data_file_path+'dataset_penn_test.pkl',penn_test_dictionary)
        print("Dataset is created and the output files are present in the dataset folder")

    elif option == '2':

        """

        Genia Corpus

        """

        genia_file_path = input("Please Enter the file Path for Genia corpus : ")
        genia_line_number =  int(input("Please Enter the Line number to check the corpus : "))

        print(" Reading the Genia dataset ")
        # Read the files from the folder
        genia_files = get_data(genia_file_path)
        # create the initial dictionary from the corpus
        genia_sentence_dict = create_genia_dataset(genia_files)
        print(len(genia_sentence_dict))

        # Statistical data of length of the word list and word counter
        genia_word_list, genia_word_counter = create_word_list(genia_sentence_dict)
        # Statistical data of lenght of the tag list and tag counter
        genia_tag_list, genia_tag_counter = create_tag_list(genia_sentence_dict)
        # Statistical data of length of the sentences
        genia_sentences_list = create_sentences_list(genia_sentence_dict)

        print("********************************************************")
        print("Initial Dictionary created")
        print("Number of Sentences : {}" .format(len(genia_sentences_list)))
        print("Number of Words : {}" .format(len(genia_word_list)))
        print("Number of Initial Tags : {}".format(len(genia_tag_counter)))
        print("Initial Tag list ")
        print(len(genia_tag_counter))
        pprint.pprint(genia_tag_counter)

        print("Initial format of the corpus with umodified tags")
        # Insert the line_number to view the line in the dataset
        print(genia_sentence_dict[genia_line_number])
        print("********************************************************")

        #intermediate storage point of reading the corpus. The corpus is written to a pickle file and then from here it is continued for further process

        genia_file_name = 'genia.pkl'
        # write the dictionary to pickle file
        write_pickle(genia_file_name,genia_sentence_dict)
        # the dictionary file read from the pickle file
        genia_data_dictionary = read_pickle(genia_file_name)
        genia_words_list , genia_words_counter = create_word_list(genia_data_dictionary)
        genia_tags_list, genia_tags_counter = create_tag_list(genia_data_dictionary)
        genia_sentences_list_new = create_sentences_list(genia_data_dictionary)

        print("********************************************************")
        print("Intermediate Pickle file created with unmodified tags")
        print("********************************************************")

        #Create a modified tag dictionary from the tagset provided in the corpus to Universal POS tags

        genia_modified_dictionary = modify_tags(genia_data_dictionary)
        genia_modified_words_list , genia_modified_words_counter = create_word_list(genia_modified_dictionary)
        genia_modified_tags_list , genia_modified_tags_counter = create_tag_list(genia_modified_dictionary)
        genia_modified_sentences_list = create_sentences_list(genia_data_dictionary)
        genia_train_dictionary , genia_test_dictionary = test_train_split(genia_modified_dictionary)

        print("********************************************************")
        print("Modified Dictionary Created")
        print("Number of Modified Tags : {}".format(len(genia_modified_tags_counter)))
        print("Modified Tag list")
        pprint.pprint(genia_modified_tags_counter)

        print("Final Format of the corpus with Modified tags")
        # Insert the line_number to view the line in the dataset
        print(genia_train_dictionary[genia_line_number])
        print("********************************************************")

        #Write output files
        # Modified tag dictionary is written in various files for the convience of the user.

        # write_txt(data_file_path+'dataset_genia.txt', genia_modified_sentences_list)
        # write_csv(data_file_path+'dataset_genia',genia_data_dictionary)
        # write_tsv(data_file_path+'dataset_genia',genia_data_dictionary)
        write_pickle(data_file_path+'dataset_genia_train.pkl',genia_train_dictionary)
        write_pickle(data_file_path+'dataset_genia_test.pkl',genia_test_dictionary)
        print("Dataset is created and the output files are present in the dataset folder")

    elif option == '3':

        """

        CONLL

        """

        conll_file_path = input("Please Enter the file Path for CoNLL corpus : ")
        conll_line_number =  int(input("Please Enter the Line number to check the corpus : "))


        print(" Reading the CoNLL dataset ")
        # Read the files from the folder
        conll_files = get_data(conll_file_path)
        # create the initial dictionary from the corpus
        conll_sentence_dict = create_conll_dataset(conll_files)
        # Statistical data - length of the word list and word counter
        conll_word_list, conll_word_counter = create_word_list(conll_sentence_dict)
        # Statistical data - lenght of the tag list and tag counter
        conll_tag_list, conll_tag_counter = create_tag_list(conll_sentence_dict)
        # Statistical data - length of the sentences
        conll_sentences_list = create_sentences_list(conll_sentence_dict)

        print("********************************************************")
        print("Initial Dictionary created")
        print("Number of Sentences : {}" .format(len(conll_sentences_list)))
        print("Number of Words : {}" .format(len(conll_word_list)))
        print("Number of Initial Tags : {}".format(len(conll_tag_counter)))
        print("Initial Tag list ")
        print(len(conll_tag_counter))
        pprint.pprint(conll_tag_counter)

        print("Initial format of the corpus with umodified tags")
        #Insert the line_number to view the line in the dataset
        print(conll_sentence_dict[conll_line_number])
        print("********************************************************")


        #intermediate storage point of reading the corpus. The corpus is written to a pickle file and then from here it is continued for further process


        conll_file_name = 'conll.pkl'
        # write the dictionary to pickle file
        write_pickle(conll_file_name,conll_sentence_dict)
        # the dictionary file read from the pickle file
        conll_data_dictionary = read_pickle(conll_file_name)
        conll_words_list , conll_words_counter = create_word_list(conll_data_dictionary)
        conll_tags_list, conll_tags_counter = create_tag_list(conll_data_dictionary)
        conll_sentences_list_new = create_sentences_list(conll_data_dictionary)

        print("********************************************************")
        print("Intermediate Pickle file created with unmodified tags")
        print("********************************************************")

        #Create a modified tag dictionary from the tagset provided in the corpus to Universal POS tags

        conll_modified_dictionary = modify_tags(conll_data_dictionary)
        conll_modified_words_list , conll_modified_words_counter = create_word_list(conll_modified_dictionary)
        conll_modified_tags_list , conll_modified_tags_counter = create_tag_list(conll_modified_dictionary)
        conll_modified_sentences_list = create_sentences_list(conll_data_dictionary)
        conll_train_dictionary , conll_test_dictionary = test_train_split(conll_modified_dictionary)

        print("********************************************************")
        print("Modified Dictionary Created")
        print("Number of Modified Tags : {}".format(len(conll_modified_tags_counter)))
        print("Modified Tag list")
        pprint.pprint(conll_modified_tags_counter)

        print("Final Format of the corpus with Modified tags")
        #Insert the line_number to view the line in the dataset
        print(conll_train_dictionary[conll_line_number])
        print("********************************************************")

        # Write Output files
        # Modified tag dictionary is written in various files for the convience of the user.

        # write_txt(data_file_path+'dataset_conll.txt', conll_modified_sentences_list)
        # write_csv(data_file_path+'dataset_conll',conll_data_dictionary)
        # write_tsv(data_file_path+'dataset_conll',conll_data_dictionary)
        write_pickle(data_file_path+'dataset_conll_train.pkl',conll_train_dictionary)
        write_pickle(data_file_path+'dataset_conll_test.pkl',conll_test_dictionary)
        print("Dataset is created and the output files are present in the dataset folder")

    elif option.lower() == 'exit':
        break

    else:
        print("Enter a valid option")
