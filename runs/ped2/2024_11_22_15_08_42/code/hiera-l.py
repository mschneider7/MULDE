from hiera import Hiera
import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_path = '/local/scratch/irr_prediction/datasets/ped2/Train'
test_path = '/local/scratch/irr_prediction/datasets/ped2/Test'

videos = [os.path.join(train_path, folder) for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))]
videos.sort()

train_batches = []
to_tensor = transforms.ToTensor()
for video in videos:
    frames = [os.path.join(video, frame) for frame in os.listdir(video) if frame.endswith('.tif')]
    frames.sort()
    rescaled_frames = []
    for frame in frames:
        img = Image.open(frame)
        img_rescaled = img.resize((224, 224))
        img_tensor = to_tensor(img_rescaled)
        rescaled_frames.append(img_tensor)
    # append 16 frames to train_batches
    for i in range(len(rescaled_frames)-15):
        batch = rescaled_frames[i:i+16]
        train_batches.append(batch)

train_batches_tensor = torch.stack([torch.stack(batch) for batch in train_batches])
# train_bathces_tensor.shape = (n_samples, n_frames, n_channels, height, width) e.g. torch.Size([105, 16, 1, 224, 224]) because grayscale
# Repeat the grayscale channel 3 times and adjust the shape
train_batches_tensor = train_batches_tensor.repeat(1, 1, 3, 1, 1)
train_batches_tensor = train_batches_tensor.permute(0, 2, 1, 3, 4)

print(f'Training batches shape: {train_batches_tensor.shape}')

test_videos = [os.path.join(test_path, folder) for folder in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, folder))]
test_videos.sort()
test_batches = []
for video in test_videos:
    frames = [os.path.join(video, frame) for frame in os.listdir(video) if frame.endswith('.tif')]
    frames.sort()
    rescaled_frames = []
    for frame in frames:
        img = Image.open(frame)
        img_rescaled = img.resize((224, 224))
        img_tensor = to_tensor(img_rescaled)
        rescaled_frames.append(img_tensor)
    # append 16 frames to test_batches
    for i in range(len(rescaled_frames)-15):
        batch = rescaled_frames[i:i+16]
        test_batches.append(batch)

test_batches_tensor = torch.stack([torch.stack(batch) for batch in test_batches])
test_batches_tensor = test_batches_tensor.repeat(1, 1, 3, 1, 1)
test_batches_tensor = test_batches_tensor.permute(0, 2, 1, 3, 4)

print(f'Testing batches shape: {test_batches_tensor.shape}')



model = Hiera.from_pretrained("facebook/hiera_large_16x224.mae_k400_ft_k400")
model.eval()
model.to(device)


# get the features
features = []

with torch.no_grad():
    for i in tqdm(range(0, len(train_batches_tensor)), total=len(train_batches_tensor), desc='Extracting training features'):
        batch = train_batches_tensor[i:i+1]

        batch = batch.to(device)
        out, intermediates = model(batch, return_intermediates=True)

        last_intermediate = intermediates[-1]  # Shape: [1, 8, 7, 7, 1152]
        
        # Flatten the tensor to combine the spatial dimensions
        flattened = last_intermediate.view(last_intermediate.size(0), -1, last_intermediate.size(-1))  # Shape: [1, 392, 1152]
        
        # Apply mean pooling to get the final 1152-dimensional feature vector
        feature = flattened.mean(dim=1)  # Shape: [1, 1152]

        feature = feature.cpu().detach()
        features.append(feature)

features_tensor = torch.cat(features, dim=0)  # Concatenate tensors along the first dimension

# Convert the tensor to a NumPy array
features_numpy = features_tensor.numpy()

np.save('ped2_train_features.npy', features_numpy)

test_features = []
with torch.no_grad():
    for i in tqdm(range(0, len(test_batches_tensor)), total=len(test_batches_tensor), desc='Extracting testing features'):
        batch = test_batches_tensor[i:i+1]

        batch = batch.to(device)
        out, intermediates = model(batch, return_intermediates=True)

        last_intermediate = intermediates[-1]  # Shape: [1, 8, 7, 7, 1152]
        
        # Flatten the tensor to combine the spatial dimensions
        flattened = last_intermediate.view(last_intermediate.size(0), -1, last_intermediate.size(-1))  # Shape: [1, 392, 1152]
        
        # Apply mean pooling to get the final 1152-dimensional feature vector
        feature = flattened.mean(dim=1)  # Shape: [1, 1152]

        feature = feature.cpu().detach()
        test_features.append(feature)

test_features_tensor = torch.cat(test_features, dim=0)  # Concatenate tensors along the first dimension

# Convert the tensor to a NumPy array
test_features_numpy = test_features_tensor.numpy()

np.save('ped2_test_features.npy', test_features_numpy)



