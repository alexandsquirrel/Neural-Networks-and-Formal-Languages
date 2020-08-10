from probe_utils import *
from dataset import *
from quantifiers import *
import sys

hidden_size = 12
exactly_three_decider = Decider(input_dimension=4, hidden_size=12)
exactly_three_decider.load_state_dict(torch.load('ckpt/best.pt'))

'''
This probe tests whether we can successfully extract the information of 
"exactly n" in the hidden representation of the model of exactly_3.
For n = 3, we expect the probe to converge nicely.
For n != 3, we expect the probe not to converge.
'''

def run_binary_classification_probe(n, verbose=True):
    classification_probe = torch.nn.Linear(hidden_size, 2)

    def exactly_n_ABs(seq):
        # TODO: perhaps, more pythonic?
        count = 0
        AB = torch.Tensor([1, 0, 0, 0])
        for vec in seq:
            if torch.equal(vec, AB):
                count += 1
        if count == n:
            return 0
        else:
            return 1

    training_inputs = make_dataset(exactly_n(n), quant_mapping, 1000, 1000)
    training_dataset = extract_hidden_states_and_labels(exactly_three_decider, training_inputs, exactly_n_ABs)
    validation_inputs = make_dataset(exactly_n(n), quant_mapping, 200, 200)
    validation_dataset = extract_hidden_states_and_labels(exactly_three_decider, validation_inputs, exactly_n_ABs)

    def h1_loss_function(x, y):
        cross_entropy_loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
        return cross_entropy_loss_function(x, y)

    train_a_probe(probe=classification_probe,
                data_training=training_dataset,
                data_validation=validation_dataset,
                num_epochs=10,
                loss_function=h1_loss_function)

    if verbose:
        print_eval_results(classification_probe, validation_dataset)


if __name__ == "__main__":
    n = int(sys.argv[1])
    run_binary_classification_probe(n, verbose=False)
