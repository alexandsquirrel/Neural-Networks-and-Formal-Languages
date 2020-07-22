from model import *
import random
from dataset import *

num_data_points = 5000
seq_length = 5

input_data = torch.zeros(num_data_points, seq_length, 2)
labels = torch.zeros(num_data_points, dtype=torch.long)

positive_sequence = torch.zeros(seq_length, 2)
for i in range(seq_length):
    positive_sequence[i][0] = 1

def generate_negative_example():
    num_ones = random.randint(1, seq_length)
    pos = random.sample(range(seq_length), num_ones)
    result = positive_sequence.clone()
    for position in pos:
        result[position][0] = 0
        result[position][1] = 1
    return result

for i in range(num_data_points):
    truth_value = random.choice([True, False])
    if truth_value == True: # bad style, but who cares :)
        input_data[i] = positive_sequence.clone()
        labels[i] = 0
    else:
        input_data[i] = generate_negative_example()
        labels[i] = 1
    #print(input_data[i], truth_value)

num_training = int(num_data_points * 0.8)
num_validation = num_data_points - num_training

t_i = input_data[:num_training]
t_l = labels[:num_training]
v_i = input_data[num_training:]
v_l = labels[num_training:]

training_data = LanguageDataset(t_i, t_l)
validation_data = LanguageDataset(v_i, v_l)

decider = Decider(input_dimension=2)
train_a_decider(model=decider, num_epochs=10, batch_size=8, data_training=training_data, data_validation=validation_data)

# external validation:
loaded_decider = Decider(input_dimension=2)
loaded_decider.load_state_dict(torch.load('checkpoint.pt'))
validation_logits = loaded_decider.forward(validation_data.data)
validation_predictions = loaded_decider.classify(validation_logits)
validation_accuracy = torch.eq(validation_predictions, validation_data.labels).double().mean()
print("External validation result: ", validation_accuracy)