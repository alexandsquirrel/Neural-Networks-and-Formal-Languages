from model import *
from quantifiers import *
from dataset import *

mapping = {'0':0, '1':1, '2':2, '3':3}
'''
zero_star = RegularLanguage(name="zero_star", chars=['0', '1'], max_length=20, 
                        regex='0*', neg_regex='[0-1]*1[0-1]*')
'''

dataset = make_dataset(every, mapping, 2, 2)
print(dataset.data, dataset.labels)
exit()

training_dataset = make_dataset(every, mapping, 1000, 1000)
validation_dataset = make_dataset(every, mapping, 200, 200)


decider = Decider()
train_a_decider(model=decider, num_epochs=100, batch_size=8,
                data_training=training_dataset, data_validation=validation_dataset)