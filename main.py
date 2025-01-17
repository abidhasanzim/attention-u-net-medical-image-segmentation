
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from drivedataset import DriveDataset
from model import AttentionUNet
from train_validate import train_model
from visualize import visualize_results

if __name__ == "__main__":
    print ("Image and mask directories")
    train_image_dir = "/Users/abid/JOB_project/segmentation/new_data_mseg/train/image"
    train_mask_dir = "/Users/abid/JOB_project/segmentation/new_data_mseg/train/mask"
    test_image_dir = "/Users/abid/JOB_project/segmentation/new_data_mseg/test/image"
    test_mask_dir = "/Users/abid/JOB_project/segmentation/new_data_mseg/test/mask"

    train_images = [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    train_masks = [os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    test_images = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    test_masks = [os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    train_images.sort()
    train_masks.sort()
    test_images.sort()
    test_masks.sort()

    train_dataset = DriveDataset(train_images, train_masks)
    test_dataset = DriveDataset(test_images, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = AttentionUNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=2)
    visualize_results(model, test_dataset)
