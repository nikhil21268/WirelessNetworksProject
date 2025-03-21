import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary

# Function to compute LoS effect
def compute_los_effect(reflectance_db, transmittance_db, distance):
    epsilon = 1e-6
    los_effect = np.exp(-(reflectance_db + transmittance_db) / (distance + epsilon))
    return los_effect

class RadioMapDataset(Dataset):
    def __init__(self, input_dir, output_dir, buildings, input_transform=None, output_transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.samples = sorted([f.split('_S')[0] + '_S' + f.split('_S')[1].split('.')[0]
                               for f in os.listdir(input_dir)
                               if f.split('_')[0] in buildings])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.samples[idx] + '.png')
        output_image_path = os.path.join(self.output_dir, self.samples[idx] + '.png')
        
        input_image = Image.open(input_image_path).convert('RGB')
        input_array = np.array(input_image)
        
        reflectance_db = input_array[:, :, 0].astype(np.float32)
        transmittance_db = input_array[:, :, 1].astype(np.float32)
        distance = input_array[:, :, 2].astype(np.float32)
        
        los_effect = compute_los_effect(reflectance_db, transmittance_db, distance)
        los_effect_image = Image.fromarray((los_effect * 255).astype(np.uint8))
        
        output_image = Image.open(output_image_path).convert('L')
        
        if self.input_transform:
            input_image = self.input_transform(input_image)
            los_effect_image = self.input_transform(los_effect_image)  # Apply same transforms to los_effect_image
            
        # Convert images to tensors
        input_image_tensor = transforms.ToTensor()(input_image)
        los_effect_image_tensor = transforms.ToTensor()(los_effect_image).unsqueeze(0)  # Ensure single channel

        # Separate channels from the input image tensor
        reflectance_image = input_image_tensor[0:1, :, :]  # Get reflectance channel
        transmittance_image = input_image_tensor[1:2, :, :]  # Get transmittance channel
        distance_image = input_image_tensor[2:3, :, :]  # Get distance channel

        # Combine all channels into a single tensor with 4 channels
        combined_input = torch.cat((reflectance_image, transmittance_image, 
                                    distance_image, los_effect_image_tensor), dim=0)
        
        if self.output_transform:
            output_image = self.output_transform(output_image)
        
        return combined_input, output_image



# Define transforms
def synchronized_transform(img, rotation_angle):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotation_angle)
    img = img.resize((256, 256))
    return img

def combined_transform(img, rotation_angle):
    if img.mode in ["RGB", "L"]:
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

# Dataset paths and buildings
input_dir = 'ICASSP2025_Dataset/Inputs/Task_1_ICASSP'
output_dir = 'ICASSP2025_Dataset/Outputs/Task_1_ICASSP'
all_buildings = [f"B{i}" for i in range(1, 26)]
np.random.seed(0)
np.random.shuffle(all_buildings)
train_buildings = all_buildings[:17]
val_buildings = all_buildings[17:21]
test_buildings = all_buildings[21:]

# Get transforms
input_transform, output_transform = get_train_transform()
val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = RadioMapDataset(input_dir, output_dir, train_buildings, input_transform=input_transform, output_transform=output_transform)
val_dataset = RadioMapDataset(input_dir, output_dir, val_buildings, input_transform=val_test_transform, output_transform=val_test_transform)
test_dataset = RadioMapDataset(input_dir, output_dir, test_buildings, input_transform=val_test_transform, output_transform=val_test_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



import torch
import torch.nn as nn
import torch.nn.functional as F
# Model definition
class WNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(WNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder path 1
        self.enc1_1 = conv_block(in_channels, 64)
        self.enc1_2 = conv_block(64, 128)
        self.enc1_3 = conv_block(128, 256)
        self.enc1_4 = conv_block(256, 512)
        
        # Encoder path 2
        self.enc2_1 = conv_block(in_channels, 64)
        self.enc2_2 = conv_block(64, 128)
        self.enc2_3 = conv_block(128, 256)
        self.enc2_4 = conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = conv_block(1024, 1024)
        
        # Decoder path
        self.upconv4 = up_conv(1024, 512)
        # Adjusted channels in dec4
        self.dec4 = conv_block(1024 + 512, 512)  # Concatenation channels adjusted
        self.upconv3 = up_conv(512, 256)
        self.dec3 = conv_block(512 + 256, 256)  # Concatenation channels adjusted
        self.upconv2 = up_conv(256, 128)
        self.dec2 = conv_block(256 + 128, 128)  # Concatenation channels adjusted
        self.upconv1 = up_conv(128, 64)
        self.dec1 = conv_block(128 + 64, 64)  # Concatenation channels adjusted
        
        # Output
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path 1
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(F.max_pool2d(enc1_1, 2))
        enc1_3 = self.enc1_3(F.max_pool2d(enc1_2, 2))
        enc1_4 = self.enc1_4(F.max_pool2d(enc1_3, 2))
        
        # Encoder path 2
        enc2_1 = self.enc2_1(x)
        enc2_2 = self.enc2_2(F.max_pool2d(enc2_1, 2))
        enc2_3 = self.enc2_3(F.max_pool2d(enc2_2, 2))
        enc2_4 = self.enc2_4(F.max_pool2d(enc2_3, 2))
        
        # Bottleneck
        bottleneck_input = torch.cat((F.max_pool2d(enc1_4, 2), F.max_pool2d(enc2_4, 2)), dim=1)
        #print(f"Bottleneck input shape: {bottleneck_input.shape}")
        bottleneck = self.bottleneck(bottleneck_input)
        
        # Decoder path
        up4 = self.upconv4(bottleneck)
        #print(f"up4 shape: {up4.shape}")
        dec4 = self.dec4(torch.cat((up4, enc1_4, enc2_4), dim=1))
        #print(f"dec4 shape: {dec4.shape}")
        
        up3 = self.upconv3(dec4)
        #print(f"up3 shape: {up3.shape}")
        dec3 = self.dec3(torch.cat((up3, enc1_3, enc2_3), dim=1))
        #print(f"dec3 shape: {dec3.shape}")
        
        up2 = self.upconv2(dec3)
        #print(f"up2 shape: {up2.shape}")
        dec2 = self.dec2(torch.cat((up2, enc1_2, enc2_2), dim=1))
        #print(f"dec2 shape: {dec2.shape}")
        
        up1 = self.upconv1(dec2)
        #print(f"up1 shape: {up1.shape}")
        dec1 = self.dec1(torch.cat((up1, enc1_1, enc2_1), dim=1))
        #print(f"dec1 shape: {dec1.shape}")
        
        # Output layer
        out = self.conv_last(dec1)
        #print(f"Output shape: {out.shape}")
        
        return out

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

# Initialize model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WNet().to(device)
summary(model, (4, 256, 256))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.5f}', "\n")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.5f}', "\n")

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epochs')
plt.legend()
plt.savefig('loss_curve_los_br_da_new.png')
plt.show()  

# Save the model
torch.save(model.state_dict(), 'wnet_model_br_los_da_new.pth')

# Load the model for inference
model.load_state_dict(torch.load('wnet_model_br_los_da_new.pth'))
model.eval()

# Test the model
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)

test_loss /= len(test_loader.dataset)
print(f'Training Loss: {epoch_loss:.5f}', "\n")
print(f'Validation Loss: {val_loss:.5f}', "\n")
print(f'Test Loss: {test_loss:.5f}', "\n")

# Save predictions vs. ground truth
def save_predictions_vs_ground_truth(dataloader, model, output_dir):
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()

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

output_dir = 'predictions_br_da_los_new'
os.makedirs(output_dir, exist_ok=True)
save_predictions_vs_ground_truth(test_loader, model, output_dir)
