from probe_utils import *
from dataset import *
from quantifiers import *

hidden_size = 12
exactly_three_decider = Decider(input_dimension=4, hidden_size=12)
exactly_three_decider.load_state_dict(torch.load('ckpt/exactly_three_10_1000_pts.pt'))

'''
Hypothesis #1: does the model keeps a binary flag (3 / not 3) ?
'''
classification_probe = torch.nn.Linear(hidden_size, 2)

def exactly_twenty_ABs(seq):
    # TODO: perhaps, more pythonic?
    count = 0
    AB = torch.Tensor([1, 0, 0, 0])
    for vec in seq:
        if torch.equal(vec, AB):
            count += 1
    if count == 20:
        return 0
    else:
        return 1

training_inputs = make_dataset(exactly_twenty, quant_mapping, 300, 300)
training_dataset = extract_hidden_states_and_labels(exactly_three_decider, training_inputs, exactly_twenty_ABs)
validation_inputs = make_dataset(exactly_twenty, quant_mapping, 100, 100)
validation_dataset = extract_hidden_states_and_labels(exactly_three_decider, validation_inputs, exactly_twenty_ABs)

def h1_loss_function(x, y):
    cross_entropy_loss_function = loss_function=torch.nn.CrossEntropyLoss(reduction='mean')
    return cross_entropy_loss_function(x, y)

train_a_probe(probe=classification_probe,
              data_training=training_dataset,
              num_epochs=40,
              loss_function=h1_loss_function)

print_eval_results(classification_probe, validation_dataset)