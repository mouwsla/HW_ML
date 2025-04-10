import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Создание тензоров 3x3
tensor1 = torch.rand(3, 3)
tensor2 = torch.rand(3, 3)
print("\nТензор 1:\n", tensor1)
print("\nТензор 2:\n", tensor2)

# Сложение тензоров
sum_tensors = tensor1 + tensor2
print("\nСумма тензоров:\n", sum_tensors)

# Поэлементное умножение
mul_tensors = tensor1 * tensor2
print("\nПоэлементное умножение:\n", mul_tensors)

# Транспонирование второго тензора
transposed_tensor2 = tensor2.T
print("\nТранспонированный тензор 2:\n", transposed_tensor2)

# Средние значения
print("\nСреднее значение тензора 1:", tensor1.mean().item())
print("Среднее значение тензора 2:", tensor2.mean().item())

# Максимальные значения
print("\nМаксимальное значение тензора 1:", tensor1.max().item())
print("Максимальное значение тензора 2:", tensor2.max().item())

# Создание нейросети для перемножения входных 2 нейрона с использованием фреймворка PyTorch

torch.manual_seed(0)
class MultModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)    
        out = self.relu(out)  
        out = self.fc2(out)  
        return out

input_size = 2
hidden_size = 100
output_size = 1
learning_rate = 0.001
num_epoch = 3000

x = torch.rand(1000, input_size) * 10
y = (x[:, 0] * x[:, 1]).reshape(-1, 1)

X_tensor = x / 10  
y_tensor = y / 100

# #Создание модели
# model = MultModule(input_size, hidden_size, output_size)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), learning_rate)

# print("\nOбучение")
# for epoch in range(num_epoch):
#     optimizer.zero_grad()
#     outputs = model(x)
#     loss = criterion(outputs, y_tensor)
#     loss.backward()
#     optimizer.step()
    
#     if (epoch+1) % 100 == 0:
#         print(f'Эпоха {epoch+1}, Loss: {loss.item():.4f}')

# torch.save(model.state_dict(), 'multiplication_model.pth')


# Загрузка модели (закомментировано обучение)
loaded_model = MultModule(input_size, hidden_size, output_size)
loaded_model.load_state_dict(torch.load('multiplication_model.pth'))
loaded_model.eval()

# Проверка загруженной модели
test_input = torch.FloatTensor([[2, 3], [5, 5], [1, 7]])
with torch.no_grad():
    loaded_pred = loaded_model(test_input)
    print("\nПроверка загруженной модели:")
    for i in range(3):
        print(f"Вход: {test_input[i].numpy()}, Ожидаемый: {test_input[i][0]*test_input[i][1]:.2f}, Предсказанный: {loaded_pred[i].item() * 100:.2f}")