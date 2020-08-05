from probe_utils import *
from dataset import *
from quantifiers import *

hidden_size = 12
exactly_three_decider = Decider(input_dimension=4, hidden_size=12)
exactly_three_decider.load_state_dict(torch.load('ckpt/best.pt'))
regression_probe = torch.nn.Linear(hidden_size, 1)

def num_AB(seq):
    # TODO: perhaps, more pythonic?
    count = 0
    AB = torch.Tensor([1, 0, 0, 0])
    for vec in seq:
        if torch.equal(vec, AB):
            count += 1
    return count

training_inputs = make_dataset(exactly_three, quant_mapping, 1010, 1010)
training_dataset = extract_hidden_states_and_labels(exactly_three_decider, training_inputs, num_AB)
validation_inputs = make_dataset(exactly_three, quant_mapping, 100, 100)
validation_dataset = extract_hidden_states_and_labels(exactly_three_decider, validation_inputs, num_AB)

def reg_loss_function(x, y):
    y = y.unsqueeze(0).transpose(0, 1).float()
    mse_loss_function = torch.nn.MSELoss()
    return mse_loss_function(x, y)

train_a_probe(probe=regression_probe,
              data_training=training_dataset,
              num_epochs=200,
              loss_function=reg_loss_function)

print_eval_results(regression_probe, validation_dataset)