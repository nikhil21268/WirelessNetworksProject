import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
from torchsummary import summary


import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class RadioMapDataset(Dataset):
    def __init__(self, input_dir, output_dir, buildings, input_transform=None, output_transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.samples = sorted([f.split('_S')[0] + '_S' + f.split('_S')[1].split('.')[0]
                               for f in os.listdir(input_dir)
                               if f.split('_')[0] in buildings])
        self.epsilon = 1e-6

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.samples[idx] + '.png')
        output_image_path = os.path.join(self.output_dir, self.samples[idx] + '.png')

        input_image = Image.open(input_image_path).convert('RGB')
        output_image = Image.open(output_image_path).convert('L')

        filename = os.path.basename(input_image_path)

        # Store the original size of the image
        original_size = input_image.size

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.output_transform:
            output_image = self.output_transform(output_image)

        # Extract RGB channels
        first_channel = input_image[0, :, :]
        second_channel = input_image[1, :, :]
        third_channel = input_image[2, :, :]

        # Calculate fourth_channel
        fourth_channel = -(first_channel + second_channel) / (third_channel + self.epsilon)
        fourth_channel = (fourth_channel - fourth_channel.min()) / (fourth_channel.max() - fourth_channel.min())
        fourth_channel = fourth_channel.unsqueeze(0)

        # Adjust the fourth channel based on frequency band
        if '_f1_' in filename:
            fourth_channel *= 3.5352884 # 10 * lembda = 10 * (3 * 10^8) / f = 10 * (3 * 10^8) / 868 MHz =  3.5352884
        elif '_f2_' in filename:
            fourth_channel *= 1.6655137 # 10 * lembda = 10 * (3 * 10^8) / f = 10 * (3 * 10^8) / 1.8 GHz =  1.6655137
        elif '_f3_' in filename:
            fourth_channel *= 0.8565499 # 10 * lembda = 10 * (3 * 10^8) / f = 10 * (3 * 10^8) / 3.5 GHz =  0.8565499


        # Concatenate the fourth_channel with the original RGB image
        input_image = torch.cat((input_image, fourth_channel), dim=0)

        return input_image, output_image, original_size, filename






# Transform functions
def synchronized_transform(img, rotation_angle,flip):
    if flip == Image.FLIP_LEFT_RIGHT:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == Image.FLIP_TOP_BOTTOM:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.rotate(rotation_angle,expand=True)
    img = img.resize((256, 256))
    return img


def combined_transform(img, rotation_angle,flip):
    if img.mode == "RGB":
        img = synchronized_transform(img, rotation_angle,flip)
    elif img.mode == "L":
        img = synchronized_transform(img, rotation_angle,flip)
    return img


def get_train_transform():
    rotation_angle = random.choice([0, 90, 180, 270])
    flip = random.choice([Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM])
    input_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Lambda(lambda img: combined_transform(img, rotation_angle,flip)),
        transforms.ToTensor()
    ])
    output_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),
        transforms.Lambda(lambda img: combined_transform(img, rotation_angle,flip)),
        transforms.ToTensor()
    ])
    return input_transform, output_transform


input_transform, output_transform = get_train_transform()

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Dataset and DataLoader
buildings = [f"B{i}" for i in range(1, 26)]

input_dir = 'ICASSP2025_Dataset/Inputs/Task_2_ICASSP_Augmented_Inputs'
output_dir = 'ICASSP2025_Dataset/Outputs/Task_2_ICASSP_Augmented_Outputs'





dataset = RadioMapDataset(input_dir, output_dir, buildings, input_transform=input_transform, output_transform=output_transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(len(dataset))




# ASPP module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_channels, kernel_size=1))

        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(x))
        conv3 = self.relu(self.conv3(x))
        conv4 = self.relu(self.conv4(x))
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(global_avg_pool, size=conv1.size()[2:], mode='bilinear', align_corners=True)

        out = torch.cat([conv1, conv2, conv3, conv4, global_avg_pool], dim=1)
        return self.conv_out(out)


# U-Net with ASPP
class UNetASPP(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNetASPP, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.aspp = ASPP(512, 1024)
        self.bottleneck = conv_block(1024, 1024)
        self.up4 = up_conv(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.up3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)
        self.up2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)
        self.up1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        aspp = self.aspp(enc4)
        bottleneck = self.bottleneck(F.max_pool2d(aspp, 2))
        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.final(dec1)


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetASPP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Model summary
summary(model, (4, 256, 256))

import time
import time
import torch
import matplotlib.pyplot as plt


# Paths to save the model and the loss plot
model_save_path = "Task2_unet_aspp_final_model_new.pth"
loss_plot_save_path = "Task2_training_loss_plot_new.png"

def compute_loss(model_output, target, original_sizes, criterion):
    total_loss = 0.0
    batch_size = model_output.size(0)

    for i in range(batch_size):
        
        height = original_sizes[0][i].item()
        width = original_sizes[1][i].item()
        original_size = (height, width)
    

        model_output_resized = F.interpolate(model_output[i:i+1], size=original_size, mode='bilinear', align_corners=True)
        target_resized = F.interpolate(target[i:i+1], size=original_size, mode='bilinear', align_corners=True)

        loss = criterion(model_output_resized, target_resized)
        total_loss += loss

    return total_loss / batch_size


loss_values = []

num_epochs = 60

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    start_time = time.time()
    
    for inputs, targets, original_sizes , _ in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = compute_loss(outputs, targets, original_sizes, criterion)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(data_loader)
    
    epoch_time = time.time() - start_time
    
    loss_values.append(avg_train_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.6f}, Time: {epoch_time :.6f}s")

 

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_train_loss
}, model_save_path)   


print(f"Model saved at: {model_save_path}")


checkpoint = torch.load(model_save_path)

model.load_state_dict(checkpoint['model_state_dict'])



plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), loss_values, marker='o', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()


plt.savefig(loss_plot_save_path)
print(f"Loss plot saved at: {loss_plot_save_path}")

plt.show()



import os

class RadioMapDataset_Test(Dataset):
    def __init__(self, input_dir, buildings, input_transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.samples = sorted([f.split('_S')[0] + '_S' + f.split('_S')[1].split('.')[0]
                               for f in os.listdir(input_dir)
                               if f.split('_')[0] in buildings])
        self.epsilon = 1e-6

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.samples[idx] + '.png')

        input_image = Image.open(input_image_path).convert('RGB')
        
        filename = os.path.basename(input_image_path)

        original_size = input_image.size

        if self.input_transform:
            input_image = self.input_transform(input_image)

        first_channel = input_image[0, :, :]
        second_channel = input_image[1, :, :]
        third_channel = input_image[2, :, :]

        fourth_channel = -(first_channel + second_channel) / (third_channel + self.epsilon)
        fourth_channel = (fourth_channel - fourth_channel.min()) / (fourth_channel.max() - fourth_channel.min())
        fourth_channel = fourth_channel.unsqueeze(0)
        
        if '_f1_' in filename:
            fourth_channel *= 3.5352884 # 10 * lembda = 10 * (3 * 10^8) / f = 10 * (3 * 10^8) / 868 MHz =  3.5352884
        elif '_f2_' in filename:
            fourth_channel *= 1.6655137 # 10 * lembda = 10 * (3 * 10^8) / f = 10 * (3 * 10^8) / 2.4 GHz =  1.25

        input_image = torch.cat((input_image, fourth_channel), dim=0)

        return input_image, original_size, filename



test_dir = "Tasks_Eval_Final/ICASSP_Test_Data/Inputs/Task_2"

# # Test loop
model.eval()

test_dataset = RadioMapDataset_Test(test_dir, buildings, input_transform = val_test_transform)

print(len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


from PIL import Image
import numpy as np
import os
import torch
import torch.nn.functional as F


def save_predictions(dataloader, model, output_dir):
    model.eval()
    with torch.no_grad():
        total_pred_time = 0
        for i, (inputs, original_size, filename) in enumerate(dataloader):
            inputs = inputs.cuda()
            
            start_time = time.time()
            
            outputs = model(inputs)
            
            prediction_time = time.time() - start_time
            
            total_pred_time = total_pred_time +  prediction_time 
            
            print(f"Prediction for Batch {i+1}: Time: {prediction_time:.6f}s")

            outputs = model(inputs)
            
            output_min = outputs.min().item()
            output_max = outputs.max().item()
           
            output_filenames =  filename

            heights = original_size[0].cpu().numpy() 
            widths = original_size[1].cpu().numpy()  
            
            for j, filename in enumerate(output_filenames): 
                height = heights[j]
                width = widths[j]
                
            
                outputs_resized = F.interpolate(
                    outputs[j:j+1],  
                    size=(width, height),
                    mode='bilinear',
                    align_corners=True
                )
            
   
                output_img = outputs_resized.squeeze().cpu().numpy() 
                
                output_img_pil = Image.fromarray((output_img * 255).astype(np.uint8))  # Scale 
                
                output_img_pil.save(os.path.join(output_dir, filename))
        print(f"Final Prediction Time for Task2 Eval Data : {total_pred_time:.6f}s")  

output_folder = "Task2_Output_Folder"
os.makedirs(output_folder, exist_ok=True)
save_predictions(test_loader, model,output_folder)
