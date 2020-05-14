import pickle
import matplotlib.pyplot as plt
import os
from helper_functions import *

# Necessary files --> CRF model files of all 6 configuration

file_path= input("Enter the Model file path :")
name_title = input("Enter the name of the Dataset")
model_files, file_names = get_model(file_path)

names = []
for name in file_names:
    text = name
    pat = re.compile(r'config_.')
    match = re.search(pat , text)
    if match :
        title = match.group(0)
        names.append(title)
#     print(title)
    


num_iterations = []
item_accuracy_float = []
sentence_accuracy_float = []
loss = []
time = []

for item in model_files:
    num_iterations_1 = []
    item_accuracy_float_1 = []
    sentence_accuracy_float_1 = []
    loss_1 = []
    time_1 = []
    for iteration in item.training_log_.iterations:
            num_iterations_1.append(iteration['num'])
            item_accuracy_float_1.append(iteration['item_accuracy_float'])
            sentence_accuracy_float_1.append(iteration['instance_accuracy_float'])
            loss_1.append(iteration['loss'])
            time_1.append(iteration['time'])
    num_iterations.append(num_iterations_1)
    item_accuracy_float.append(item_accuracy_float_1)
    sentence_accuracy_float.append(sentence_accuracy_float_1)
    loss.append(loss_1)
    time.append(time_1)

fig, axs = plt.subplots(2, 2,figsize=(10,10))
fig.suptitle("{}".format(name_title),fontweight="bold", size=20,color = 'b')
l1 = axs[0, 0].plot(num_iterations[0],loss[0])
l2 = axs[0, 0].plot(num_iterations[1],loss[1])
l3 = axs[0, 0].plot(num_iterations[2],loss[2])
l4 = axs[0, 0].plot(num_iterations[3],loss[3])
l5 = axs[0, 0].plot(num_iterations[4],loss[4])
l6 = axs[0, 0].plot(num_iterations[5],loss[5])
l7 = axs[0, 0].plot(num_iterations[6],loss[6])
axs[0, 0].legend([x for x in names],loc="upper right")
axs[0, 0].set_title('Loss SGD',fontweight="bold", size=15,color = 'w')
axs[0,0].set(ylabel = 'Loss')
axs[0, 1].plot(num_iterations[0],time[0])
axs[0, 1].plot(num_iterations[1],time[1])
axs[0, 1].plot(num_iterations[2],time[2])
axs[0, 1].plot(num_iterations[3],time[3])
axs[0, 1].plot(num_iterations[4],time[4])
axs[0, 1].plot(num_iterations[5],time[5])
axs[0, 1].plot(num_iterations[6],time[6])
axs[0, 1].legend([x for x in names],loc="upper right")
axs[0, 1].set_title('Time per iteration',fontweight="bold", size=15,color = 'w')
axs[0,1].set(ylabel = 'Time sec')
axs[0,1].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axs[1, 0].plot(num_iterations[0],item_accuracy_float[0])
axs[1, 0].plot(num_iterations[1],item_accuracy_float[1])
axs[1, 0].plot(num_iterations[2],item_accuracy_float[2])
axs[1, 0].plot(num_iterations[3],item_accuracy_float[3])
axs[1, 0].plot(num_iterations[4],item_accuracy_float[4])
axs[1, 0].plot(num_iterations[5],item_accuracy_float[5])
axs[1, 0].plot(num_iterations[6],item_accuracy_float[6])
axs[1, 0].legend([x for x in names])
axs[1,0].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axs[1, 0].set_title('Word Accuracy',fontweight="bold", size=15,color = 'w')
axs[1,0].set(xlabel = 'Iterations' , ylabel = 'Accuracy')
axs[1, 1].plot(num_iterations[0],sentence_accuracy_float[0])
axs[1, 1].plot(num_iterations[1],sentence_accuracy_float[1])
axs[1, 1].plot(num_iterations[2],sentence_accuracy_float[2])
axs[1, 1].plot(num_iterations[3],sentence_accuracy_float[3])
axs[1, 1].plot(num_iterations[4],sentence_accuracy_float[4])
axs[1, 1].plot(num_iterations[5],sentence_accuracy_float[5])
axs[1, 1].plot(num_iterations[6],sentence_accuracy_float[6])
axs[1, 1].legend([x for x in names],loc="upper right")
axs[1,1].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
axs[1, 1].set_title('Sentence Accuracy',fontweight="bold", size=15 ,color = 'w')
axs[1,1].set(xlabel = 'Iterations' , ylabel = 'Accuracy')

plt.show()


