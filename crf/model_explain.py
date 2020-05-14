import eli5
from helper_functions import *

file_path= input("Enter the Model file path :")
# name_title = input("Enter the name of the Dataset  :  ")
model_files, file_names = get_model(file_path)
i = 0
for i in range(len(model_files)): 
	explain = eli5.explain_weights(model_files[i] , top = 30 )
	x = eli5.format_as_html(explain)
	f = open('{}.html'.format(file_names[i]),'w')
	f.write(x)
	i += 1