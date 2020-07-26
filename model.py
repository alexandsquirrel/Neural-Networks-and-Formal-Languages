import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *


class Decider(nn.Module):

    def __init__(self, input_dimension=4, hidden_size=12, num_layers=2):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_dimension,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers)
        # TODO: add dropout, potentially.
        self.classifier = nn.Linear(self.hidden_size, 2) # 2 stands for accept/reject
        self.softmax = nn.Softmax(1) # not an actual layer
        # TODO: move to training
        self.loss_calc = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def forward(self, sequences, seq_lengths):
        idx_last_output = seq_lengths - 1
        # sequence: [batch_size, max_len, input_dimension]
        # input to lstm: [max_len, batch_size, input_dimension]
        all_outputs, _ = self.lstm(torch.transpose(sequences, 0, 1))
        # all_outputs: [max_len, batch_size, hidden_size=lstm_output_dimensions]
        final_outputs = all_outputs[idx_last_output, torch.arange(len(sequences), out=torch.LongTensor()), :]
        # final_outputs: [batch_size, hidden_size]
        logits = self.classifier(final_outputs)
        # logits: [batch_size, 2]
        return logits

    def classify(self, logits):
        # logits: [batch_size, 2]
        probs = self.softmax(logits)
        categories = torch.argmax(probs, dim=1)
        return categories
    
    def loss(self, actual, target):
        # actual: [batch_size, 2]
        # target: [batch_size] where each value denotes the # of class
        # each row of actual should be an unnormalized logit.
        return self.loss_calc(actual, target)


def train_a_decider(model, num_epochs: int, batch_size: int, 
                    data_training: LanguageDataset, data_validation: LanguageDataset,
                    save_file='checkpoint.pt'):
    # data_training: [num_training_data_pts, max_len, input_dimension]
    # data_validation: [num_validation_data_pts, max_len, input_dimension]
    current_epoch = 0
    best_model = dict()
    best_model_accuracy = 0.0
    while current_epoch < num_epochs:
        current_epoch += 1
        training_loss = 0
        # STEP (1): split the training data and the validation data (?) into small batches.
        dataloader = torch.utils.data.DataLoader(dataset=data_training,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        # STEP (2): train and print accuracy of validation along the way.
        for batch, labels, seq_lengths in dataloader.__iter__():
            logits = model.forward(batch, seq_lengths)
            loss = model.loss(logits, labels)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            training_loss += loss
        
        # STEP (3): validate this epoch
        validation_logits = model.forward(data_validation.data, data_validation.lengths)
        validation_predictions = model.classify(validation_logits)
        validation_accuracy = torch.eq(validation_predictions, data_validation.labels).double().mean()
        print("In epoch %d, training_loss = %f, validation accuracy = %f" % (current_epoch, training_loss, validation_accuracy))
        if validation_accuracy > best_model_accuracy:
            best_model_accuracy = validation_accuracy
            best_model = model.state_dict()

    # STEP (4): freeze the weights into disk.
    torch.save(best_model, save_file)