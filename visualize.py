
import torch
import matplotlib.pyplot as plt

def visualize_results(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    images, masks = dataset[0]
    images = images.unsqueeze(0).to(device)
    masks = masks.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_masks = model(images)
        predicted_masks = torch.sigmoid(predicted_masks).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(images.cpu().squeeze().permute(1, 2, 0))

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(masks.cpu().squeeze(), cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_masks.squeeze(), cmap="gray")
    plt.show()
