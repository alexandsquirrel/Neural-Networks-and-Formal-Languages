from dataset import *
from quantifiers import *
from model import *

'''
This script is to test whether the length of a sequence has any impact
on the decider's decision. We aim to find this out by pretty-printing
the validation results of a validation dataset and trying to find whether
wrong classification results are correlated with abnormal sequence lengths.
'''

hidden_size = 12
exactly_three_decider = Decider(input_dimension=4, hidden_size=12)
exactly_three_decider.load_state_dict(torch.load('ckpt/exactly_three_10_1000_pts.pt'))
validation_dataset = make_dataset(exactly_three, quant_mapping, 100, 100)

logits = exactly_three_decider(validation_dataset.data, validation_dataset.lengths)
classification_results = exactly_three_decider.classify(logits)

for idx, (seq, label, seq_length) in enumerate(validation_dataset):
    if classification_results[idx] == label:
        result = "Correct decision,"
    elif classification_results[idx] == 0:
        result = "Negative sequence wrongly classified as positive,"
    else:
        result = "Positive sequence wrongly classified as negative,"
    print(result, "len =", seq_length)