import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

class RadioMapDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.samples = sorted([f.split('_S')[0] + '_S' + f.split('_S')[1].split('.')[0] 
                               for f in os.listdir(input_dir)])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.samples[idx] + '.png')
        output_image_path = os.path.join(self.output_dir, self.samples[idx] + '.png')
        
        input_image = Image.open(input_image_path).convert('RGB')
        output_image = Image.open(output_image_path).convert('L')  # Grayscale
        
        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)
        
        return input_image, output_image

# Set up the transform with data augmentation
transform = transforms.Compose([
    
    transforms.Resize((256, 256)),  # Resize images to 256x256
    #transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor()  # Convert the image to a tensor
    
    # transforms.Resize((256, 256)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=(-15, 15)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    # transforms.ToTensor()
])

# Directories for input and output images
input_dir = '/home/shamik/ICASSP/ICASSP2025_Dataset/Inputs/Task_1_ICASSP'
output_dir = '/home/shamik/ICASSP/ICASSP2025_Dataset/Outputs/Task_1_ICASSP'

# Load the full dataset
full_dataset = RadioMapDataset(input_dir, output_dir, transform=transform)
print("No of Images in Dataset ", full_dataset)

# Define the sizes for train, validation, and test sets
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoader for each dataset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Print sizes of the datasets
print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = up_conv(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)
        
        # Output
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path
        up4 = self.upconv4(bottleneck)
        
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        
        # Output layer
        out = self.conv_last(dec1)
        
        return out

# Initialize the model
model = UNet(in_channels=3, out_channels=1).cuda()

# # Define the loss function and optimizer

# Check input and output sizes by passing a dummy input through the model
dummy_input = torch.randn(1, 3, 256, 256).cuda()  # Batch size of 1, 3 channels, 256x256 image
dummy_output = model(dummy_input)

print(f'Input size: {dummy_input.size()}')
print(f'Output size: {dummy_output.size()}')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


from torchsummary import summary

# Assuming 'model' is already defined and on the GPU
summary(model, (3, 256, 256))
print(summary(model, (3, 256, 256)))

import matplotlib.pyplot as plt

# Number of epochs
num_epochs = 200

# Lists to store loss values
train_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Training phase
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # # Print size of actual input and output
        # print(f'Batch input size: {inputs.size()}')
        # print(f'Batch target size: {targets.size()}')

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # print(f'Batch output size: {outputs.size()}')  # Print size of output
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.5f}',"\n")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # # Print size of actual input and output
            # print(f'Batch input size: {inputs.size()}')
            # print(f'Batch target size: {targets.size()}')
            
            outputs = model(inputs)
            
            # print(f'Batch output size: {outputs.size()}')  # Print size of output
            
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.5f}',"\n")

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
# Save the plot to the same folder as the script
plt.savefig('loss_curve.png')
plt.show()  


# Save the model
torch.save(model.state_dict(), 'unet_model.pth')

# # Load the model for inference
# model = UNet().cuda()
model.load_state_dict(torch.load('unet_model.pth'))
model.eval()


# Test the model
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        # print(f'Test batch input size: {inputs.size()}')
        # print(f'Test batch target size: {targets.size()}')
        # print(f'Test batch output size: {outputs.size()}')  # Print size of output
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)

test_loss /= len(test_loader.dataset)

from torchsummary import summary

# Assuming 'model' is already defined and on the GPU
summary(model, (3, 256, 256))
print(summary(model, (3, 256, 256)))

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

output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
save_predictions_vs_ground_truth(test_loader, model, output_dir)





