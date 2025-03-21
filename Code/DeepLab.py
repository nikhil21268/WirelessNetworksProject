import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the dataset class
class RadioMapDataset(Dataset):
    def __init__(self, input_dir, output_dir, buildings, input_transform=None, output_transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.samples = sorted([f.split('_S')[0] + '_S' + f.split('_S')[1].split('.')[0] 
                               for f in os.listdir(input_dir)
                               if f.split('_')[0] in buildings])
        self.epsilon = 1e-6  # Small value to avoid division by zero
    
    def __len__(self):
        return len(self.samples)
    
    def get_filenames(self):
        return self.samples
    
    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.samples[idx] + '.png')
        output_image_path = os.path.join(self.output_dir, self.samples[idx] + '.png')
        
        input_image = Image.open(input_image_path).convert('RGB')
        output_image = Image.open(output_image_path).convert('L')  # Grayscale
        
        if self.input_transform:
            input_image = self.input_transform(input_image)
            
        if self.output_transform:
            output_image = self.output_transform(output_image)
        
        return input_image, output_image

# Define synchronized transform that takes a rotation angle and applies the same transformation to both input and output
def synchronized_transform(img, rotation_angle):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotation_angle)
    img = img.resize((256, 256))  
    return img

def combined_transform(img, rotation_angle):
    if img.mode == "RGB":  # Input (RGB) image
        img = synchronized_transform(img, rotation_angle)
    elif img.mode == "L":  # Output (grayscale) image
        img = synchronized_transform(img, rotation_angle)
    
    return img

def get_train_transform():
    rotation_angle = random.choice([0, 90, 180, 270])
    input_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Lambda(lambda img: combined_transform(img, rotation_angle)),
        transforms.ToTensor()
    ])
    
    output_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),
        transforms.Lambda(lambda img: combined_transform(img, rotation_angle)),
        transforms.ToTensor()
    ])
    
    return input_transform, output_transform

input_transform, output_transform = get_train_transform()

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

all_buildings = [f"B{i}" for i in range(1, 26)]
np.random.seed(0)
np.random.shuffle(all_buildings)
train_buildings = all_buildings[:17]
val_buildings = all_buildings[17:21]
test_buildings = all_buildings[21:]

input_dir = 'ICASSP2025_Dataset/Inputs/Task_1_ICASSP'
output_dir = 'ICASSP2025_Dataset/Outputs/Task_1_ICASSP'

train_dataset = RadioMapDataset(input_dir, output_dir, train_buildings, input_transform=input_transform, output_transform=output_transform)
val_dataset = RadioMapDataset(input_dir, output_dir, val_buildings, input_transform=val_test_transform, output_transform=val_test_transform)
test_dataset = RadioMapDataset(input_dir, output_dir, test_buildings, input_transform=val_test_transform, output_transform=val_test_transform)

print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

print(f'Train buildings: {train_buildings}')
print(f'Validation buildings: {val_buildings}')
print(f'Test buildings: {test_buildings}')

print('Training Dataset Filenames:')
for filename in train_dataset.get_filenames():
    print(filename)

print('\nValidation Dataset Filenames:')
for filename in val_dataset.get_filenames():
    print(filename)

print('\nTest Dataset Filenames:')
for filename in test_dataset.get_filenames():
    print(filename)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define a model with GroupNorm replacing BatchNorm
class DeepLabV3WithGroupNorm(nn.Module):
    def __init__(self):
        super(DeepLabV3WithGroupNorm, self).__init__()
        # Load the pre-trained DeepLabV3 model
        self.deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=True)
        
        # Replace BatchNorm with GroupNorm
        def replace_batchnorm_with_groupnorm(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    num_features = child.num_features
                    module.add_module(name, nn.GroupNorm(num_groups=32, num_channels=num_features))
                else:
                    replace_batchnorm_with_groupnorm(child)
        
        replace_batchnorm_with_groupnorm(self.deeplabv3)

        # Modify classifier layer to output 1 class
        self.deeplabv3.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))

    def forward(self, x):
        return self.deeplabv3(x)

# Initialize the model
model = DeepLabV3WithGroupNorm().cuda()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check input and output sizes by passing a dummy input through the model
dummy_input = torch.randn(1, 3, 256, 256).cuda()
dummy_output = model(dummy_input)['out']
print(f'Input size: {dummy_input.size()}')
print(f'Output size: {dummy_output.size()}')

# Training loop
num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.5f}',"\n")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)['out']
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.5f}',"\n")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
plt.savefig('loss_curve_deeplab.png')
plt.show()

torch.save(model.state_dict(), 'deeplab_model.pth')

model.load_state_dict(torch.load('deeplab_model.pth'))
model.eval()

test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)['out']
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)

test_loss /= len(test_loader.dataset)

print(f'Test Loss: {test_loss:.5f}')
from torchsummary import summary

# # Assuming 'model' is already defined and on the GPU
# summary(model, ( 3, 256, 256))
# print(summary(model, (1, 3, 256, 256)))

print(f'Training Loss: {epoch_loss:.5f}',"\n")
print(f'Validation Loss: {val_loss:.5f}',"\n")
print(f'Test Loss: {test_loss:.5f}',"\n")
 
import numpy as np

# Save predictions vs. ground truth
def save_predictions_vs_ground_truth(dataloader, model, output_dir):
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            # Convert tensors to numpy arrays and denormalize if needed
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()

            # Save images
            for j in range(inputs_np.shape[0]):
                input_img = np.transpose(inputs_np[j], (1, 2, 0))
                target_img = targets_np[j, 0]  # Grayscale image
                output_img = outputs_np[j, 0]  # Grayscale image

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.title('Input')
                plt.imshow(input_img)
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title('Target')
                plt.imshow(target_img, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title('Output')
                plt.imshow(output_img, cmap='gray')
                plt.axis('off')

                plt.savefig(os.path.join(output_dir, f'pred_vs_gt_{i*16 + j}.png'))
                plt.close()

output_dir = 'predictions_br_da_DeepLab'
os.makedirs(output_dir, exist_ok=True)
save_predictions_vs_ground_truth(test_loader, model, output_dir)





