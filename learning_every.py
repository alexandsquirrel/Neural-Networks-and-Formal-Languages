from model import *
from quantifiers import *
from dataset import *

mapping = {'0':0, '1':1, '2':2, '3':3}
zero_star_mapping = {'0':0, '1':1}
zero_star = RegularLanguage(name="zero_star", chars=['0', '1'], max_length=20, 
                        regex='0*', neg_regex='[0-1]*1[0-1]*')

'''
dataset = make_dataset(zero_star, zero_star_mapping, 2, 2)
print(dataset.data, dataset.labels, dataset.lengths)
exit()
'''
'''
training_dataset = make_dataset(zero_star, zero_star_mapping, 1000, 1000)
validation_dataset = make_dataset(zero_star, zero_star_mapping, 200, 200)
'''

training_dataset = make_dataset(exactly_three, mapping, 1000, 1000)
validation_dataset = make_dataset(exactly_three, mapping, 200, 200)

'''
training_dataset = make_dataset(every, mapping, 1000, 1000)
validation_dataset = make_dataset(every, mapping, 200, 200)
'''


decider = Decider(input_dimension=4)
train_a_decider(model=decider, num_epochs=10, batch_size=8,
                data_training=training_dataset, data_validation=validation_dataset)