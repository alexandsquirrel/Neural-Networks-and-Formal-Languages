from model import *
import random
from dataset import *
from quantifiers import *


mapping = {'0':0, '1':1, '2':2, '3':3}
training_dataset = make_dataset(exactly_three, mapping, 300, 300)
validation_dataset = make_dataset(exactly_three, mapping, 100, 100)
#print("Actual value: ", validation_dataset[595][0].shape)

##################
# Probing:


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(12, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x

###

loaded_decider = Decider(input_dimension=4)
loaded_decider.load_state_dict(torch.load('checkpoint.pt'))

model1 = loaded_decider
model1.eval()

logits = loaded_decider.forward(training_dataset.data, seq_lengths=training_dataset.lengths)
# Why do we need to keep the logits? Or calling forward once is all we want?

hidden_rep = model1.hidden_rep.detach()
print(hidden_rep.shape)


#   Probe Demo:

model_regression = LinearModel()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_regression.parameters(), lr=0.001)
model_regression.train()

def get_label(index, dataset):
    AB = torch.Tensor([1, 0, 0, 0])
    if dataset[index][1] == 0:
        label = 3
    else:
        label = 0
        for i in range(dataset[index][2]):
            if torch.equal(dataset[index][0][i], AB):
                label += 1
    return label

for epoch in range(6):
    sum_loss = 0
    for i in range(600):
        layer = hidden_rep[i,:] # change to [i] ?
        #print(layer[0:5])  # test
        # print(layer.shape)

        x_data = torch.Tensor(layer)
        label = get_label(i, training_dataset)
        #print(i, label)
        y_data = torch.Tensor([label])  # Exactly 'n'

        optimizer.zero_grad()
        # Forward pass
        y_pred = model_regression(x_data)
        # Compute Loss
        loss = loss_function(y_pred, y_data)
        sum_loss += loss
        # Backward pass
        loss.backward()
        optimizer.step()
    print("in epoch %d, loss = %f" % (epoch, sum_loss / 600))


#    todo: make a loop for testing

print('\n')

model_regression.eval()
loaded_decider.forward(validation_dataset.data, seq_lengths=validation_dataset.lengths)
hidden_rep = model1.hidden_rep

for idx in range(200):
    new_x = torch.Tensor(hidden_rep[idx,:])
    y_pred = model_regression(new_x)
    print("Predicted value: %f, Actual value: %d" % (y_pred, get_label(idx, validation_dataset)))
#print("Actual value: ", validation_dataset[595][0])
