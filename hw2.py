import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size_1,hidden_size_2,hidden_size_3, output_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(32, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)  
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.leaky_relu(out)
        out = self.output(out)     
        out = self.sigmoid(out)
        return out
    

input_size = 5
hidden_size_1 = 64
hidden_size_2 = 32
hidden_size_3 = 16
output_size = 1
learning_rate = 0.001
num_epoch = 1000


x= torch.randn(1000, 5)
y = (x.sum(dim=1) > 0).float().view(-1, 1)


def train_model(optimizer_type='adam', num_epoch = 1000):
    model = BinaryClassifier(input_size, hidden_size_1,hidden_size_2,hidden_size_3, output_size)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), learning_rate) if optimizer_type == 'adam' else \
                optim.SGD(model.parameters(), learning_rate)
    
    losses = []
    for _ in range(num_epoch):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"{optimizer_type.upper()} final loss: {loss.item():.4f}")
    return losses


loss_adam = train_model('adam')
loss_sgd = train_model('sgd')

plt.plot(loss_adam, label='Adam', color='blue')
plt.plot(loss_sgd, label='SGD', color='orange')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Сравнение')
plt.legend()
plt.grid(True)
plt.show()
