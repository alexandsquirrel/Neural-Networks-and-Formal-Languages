from probe_utils import *
from dataset import *
from quantifiers import *

hidden_size = 12
exactly_three_decider = Decider(input_dimension=4, hidden_size=12)
exactly_three_decider.load_state_dict(torch.load('ckpt/exactly_three_10_1000_pts.pt'))
regression_probe = torch.nn.Linear(hidden_size, 1)

def len_seq(seq):
    # TODO: perhaps, more pythonic?
    count = 0
    empty = torch.Tensor([0, 0, 0, 0])
    for vec in seq:
        if not torch.equal(vec, empty):
            count += 1
        else:
            break
    return count

training_inputs = make_dataset(exactly_three, quant_mapping, 300, 300)
training_dataset = extract_hidden_states_and_labels(exactly_three_decider, training_inputs, len_seq)
validation_inputs = make_dataset(exactly_three, quant_mapping, 100, 100)
validation_dataset = extract_hidden_states_and_labels(exactly_three_decider, validation_inputs, len_seq)

def reg_loss_function(x, y):
    y = y.unsqueeze(0).transpose(0, 1).float()
    mse_loss_function = torch.nn.MSELoss()
    return mse_loss_function(x, y)

train_a_probe(probe=regression_probe,
              data_training=training_dataset,
              num_epochs=100,
              loss_function=reg_loss_function)

print_eval_results(regression_probe, validation_dataset)