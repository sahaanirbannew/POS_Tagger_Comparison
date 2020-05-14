from corpus_reader import * 
import pickle 
import pprint

file_train = input("Enter the file path of the train dictionary")
file_test = input("Enter the file path of the test dictionary")

train_dictionary = pickle.load(open(file_train,'rb'))
test_dictionary = pickle.load(open(file_test,'rb'))

train_tags_list , train_tags_counter = create_tag_list(train_dictionary)
test_tags_list , test_tags_counter = create_tag_list(test_dictionary)

print("Train Tags Counter")
pprint.pprint(train_tags_counter)
print("#"*50)
print("Test Tags Counter")
pprint.pprint(test_tags_counter)




