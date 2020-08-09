from probe_utils import *
from dataset import *
from quantifiers import *
import sys

hidden_size = 12
exactly_three_decider = Decider(input_dimension=4, hidden_size=12)
exactly_three_decider.load_state_dict(torch.load('ckpt/best.pt'))

'''
This is to test whether we can successfully extract the information of 
"exactly n" (with n\ne 3) in the hidden representation of the model of exactly_3.
We expect this to fail.
'''

if __name__ == "__main__":
    n = int(sys.argv[1])

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

    #print_eval_results(classification_probe, validation_dataset)
