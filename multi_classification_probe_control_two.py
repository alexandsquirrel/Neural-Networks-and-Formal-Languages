from probe_utils import *
from dataset import *
from quantifiers import *

hidden_size = 12
exactly_three_decider = Decider(input_dimension=4, hidden_size=12)
exactly_three_decider.load_state_dict(torch.load('ckpt/best.pt'))

'''
Hypothesis #1: does the model do a multi-way classification?
zero - one - two - three - more_than_three
Note: this is by chance also a DFA probe, on this task.

YET ANOTHER CONTROL TEST FOR ABOVE.
Note: we somehow expect n < 3 to converge nicely, for the clustering argument.
'''

'''
Let's probe for exactly_2: 4 categories in total, analogously.
'''
classification_probe = torch.nn.Linear(hidden_size, 4)

def num_AB(seq):
    # TODO: perhaps, more pythonic?
    count = 0
    AB = torch.Tensor([1, 0, 0, 0])
    for vec in seq:
        if torch.equal(vec, AB):
            count += 1
    return min(count, 3)

training_inputs = make_dataset(exactly_two, quant_mapping, 1000, 1000)
training_dataset = extract_hidden_states_and_labels(exactly_three_decider, training_inputs, num_AB)
validation_inputs = make_dataset(exactly_two, quant_mapping, 100, 100)
validation_dataset = extract_hidden_states_and_labels(exactly_three_decider, validation_inputs, num_AB)

def h1_loss_function(x, y):
    cross_entropy_loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    return cross_entropy_loss_function(x, y)

train_a_probe(probe=classification_probe,
              data_training=training_dataset,
              data_validation=validation_dataset,
              num_epochs=40,
              loss_function=h1_loss_function)

print_eval_results(classification_probe, validation_dataset)
