import torch
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

root = "./Data_10"
model_save_path = './LearnModel.pth'
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 10

train_set = CIFAR10(train=True, transform=transformation, root=root, download=True)
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = CIFAR10(train=False, transform=transformation, root=root, download=True)
test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

class NewImgModel(nn.Module):
    def __init__(self):
        super(NewImgModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)

        self.fc = nn.Linear(24 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 24 * 16 * 16)
        return self.fc(x)


def test_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data_loader:
            outputs = model(images)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


model = NewImgModel()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
num_epochs = 10
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss_value = loss(outputs, labels)
        loss_value.backward()
        optimizer.step()

    accuracy = test_accuracy(model)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), model_save_path)

    print(f"Epoch {epoch+1}; Accuracy: {accuracy:.2f}%")


load_model = NewImgModel()
load_model.load_state_dict(torch.load(model_save_path))
load_model.eval()


images, true_labels = next(iter(test_data_loader))
images, true_labels = images[:20], true_labels[:20]

outputs = load_model(images)
predicted_labels = torch.max(outputs, 1)[1]

def print_labels(title, labels):
    print(title, end=' ')
    for i in range(len(labels)):
        print(classes[labels[i]], end=' ')
    print()

print_labels("True labels:     ", true_labels)
print_labels("Predicted labels:", predicted_labels)


img_grid = torchvision.utils.make_grid(images, nrow=10)
img_grid = img_grid / 2 + 0.5 
plt.figure(figsize=(15, 4))
plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
plt.axis('off')
plt.title("Result")
plt.show()


correct_20 = (predicted_labels == true_labels).sum().item()
accuracy_20 = correct_20 / 20 * 100
print(f"Accuracy on 20 test images: {accuracy_20:.2f}%")