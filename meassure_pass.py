from datetime import datetime  
import time
import pytz
import torch
import torch.optim as optim
from vit_loader import vit_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
# Training settings
epochs = 20 #20
lr = 3e-6 #3e-5
gamma = 0.7 #0.7

my_timezone = pytz.timezone('Europe/Berlin')  
now = datetime.now(my_timezone)  
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

# Assume input and target are your input and target data
input = torch.randn(32, 3, 32, 32)
target = torch.randint(0, 10, (32,))

model_original_unstructured = torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-07-04_17-18-01.pth') #  Original Unstructured Pruned Model: torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/model_2023-07-04_17-18-01.pth'), Structured Pruned Model: torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/structural_pruned_2023-07-04_20-13-41.pth')
model_structured = torch.load('/home/paperspace/Desktop/MT_Bachelor_ViT_Rationals_Pruning/pruned_models/structural_pruned_2023-07-04_20-13-41.pth')

#model = model_original_unstructured
model = model_structured

# Create a loss function and an optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Record the start time
start_time = time.time()
output = model(input)
loss = criterion(output, target)
forward_time = time.time()
optimizer.zero_grad()
loss.backward()
optimizer.step()
end_time = time.time()

# Print the times
print(f"Forward pass time: {forward_time - start_time} seconds")
print(f"Backward pass time: {end_time - forward_time} seconds")

