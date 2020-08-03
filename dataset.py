import torch
from language import Language

class LanguageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, lengths):
        self.data = data
        self.labels = labels
        self.lengths = lengths
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.lengths[index]

    def __len__(self):
        return len(self.data)


def strings_to_tensor(strings: set, seq_length: int, mapping: dict):
    tensor = torch.zeros(len(strings), seq_length, len(mapping))
    lengths = torch.zeros(len(strings), dtype=torch.long)
    for i, string in enumerate(strings):
        lengths[i] = len(string)
        for j, char in enumerate(string):
            tensor[i][j][mapping[char]] = 1
    tensor.requires_grad = True
    return tensor, lengths

def make_dataset(language: Language, mapping: dict, num_positive: int, num_negative: int):
    positive_strings = language.generate_batch_data(num_datapoints=num_positive, truth_value=True)
    negative_strings = language.generate_batch_data(num_datapoints=num_negative, truth_value=False)
    # TODO: fix language._max_length
    max_length = max([len(string) for string in positive_strings + negative_strings])
    # print(max_length)
    positive_tensor, positive_lengths = strings_to_tensor(positive_strings, max_length, mapping)
    positive_labels = torch.zeros(len(positive_strings), dtype=torch.long)
    negative_tensor, negative_lengths = strings_to_tensor(negative_strings, max_length, mapping)
    negative_labels = torch.ones(len(negative_strings), dtype=torch.long)
    all_tensors = torch.cat((positive_tensor, negative_tensor))
    all_labels = torch.cat((positive_labels, negative_labels))
    all_lengths = torch.cat((positive_lengths, negative_lengths))
    return LanguageDataset(all_tensors, all_labels, all_lengths) # unshuffled, must be shuffled at use
