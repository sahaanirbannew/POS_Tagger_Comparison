from train_conf import *

configuration = {0 : 'Word + Morphological features',
                 1 : 'Word feature',
                 2 : 'Word + Orthography + Morphology features',
                 3 : 'Word + Orthography features',
                 4 : 'Morphology feature',
                 5 : 'Orthography + Morphology features',
                 6 : 'Orthography feature'}

num = int(input("Enter the configuration number for drf "))
print(configuration[num])
train_configuration(num)

