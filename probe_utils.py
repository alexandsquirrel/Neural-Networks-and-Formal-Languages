import torch
from model import *

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

def train_a_probe(probe, data_training, num_epochs, loss_function, batch_size=4):
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    current_epoch = 0
    while current_epoch < num_epochs:
        current_epoch += 1
        training_loss = 0
        dataloader = torch.utils.data.DataLoader(dataset=data_training,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        for hidden_reps, labels in dataloader:
            optimizer.zero_grad()
            preds = probe.forward(hidden_reps)
            loss = loss_function(preds, labels)
            training_loss += loss
            loss.backward()
            optimizer.step()
        print("in epoch %d, loss = %f" % (current_epoch, training_loss))

def print_eval_results(probe, data_eval):
    probe.eval()
    for hidden_rep, label in data_eval:
        pred = probe(hidden_rep)
        print(pred,"vs.", label)

def extract_hidden_states_and_labels(decider, input_data, label_of):
    decider.eval()
    # TODO: support entire sequence
    decider(input_data.data, seq_lengths=input_data.lengths)
    hidden_reps = decider.hidden_rep.detach()
    labels = [label_of(sequence) for sequence in input_data.data]
    return ProbeDataset(hidden_reps, labels)