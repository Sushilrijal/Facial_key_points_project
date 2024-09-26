import torch
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def train_batch(imgs, kps, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    # forward Pass
    kps_pred = model(imgs)
    loss = criterion(kps_pred, kps)

    # backward Pass
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def validation_batch(imgs, kps, model, criterion):
    model.eval()

    kps_pred = model(imgs)
    loss = criterion(kps_pred, kps)
    return loss


def train(n_epoch, train_dataloader, test_dataloader, model, criterion, optimizer):
    train_loss = []
    test_loss = []
    for epoch in range(1, n_epoch + 1):
        epoch_train_loss, epoch_test_loss = 0, 0

        # train
        for images, kps in tqdm(
            train_dataloader, desc=f"Training {epoch} of {n_epoch}"
        ):
            # images, kps =
            loss = train_batch(images, kps, model, criterion, optimizer)
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_dataloader)
        train_loss.append(epoch_train_loss)

        # validation
        for images, kps in tqdm(test_dataloader, desc="validation"):
            loss = validation_batch(images, kps, model, criterion)
            epoch_test_loss += loss.item()
        epoch_test_loss /= len(test_dataloader)
        test_loss.append(epoch_test_loss)


        print(
            f"Epoch {epoch} of {n_epoch}: Training Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}"
        )
    return train_loss, test_loss


def plot_curve(train_loss, test_loss, train_curve_path):
    epochs = np.arange(len(train_loss))

    plt.figure()
    plt.plot(epochs, train_loss, "b", label="Training Loss")
    plt.plot(epochs, test_loss, "r", label="Test Loss")
    plt.title("Training and Test Loss Curve Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("SmoothL1 Loss")
    plt.legend()
    plt.show()
    plt.savefig(train_curve_path)


def load_img(img_path, model_input_size, device):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # [r,g,b]
        std=[0.229, 0.224, 0.225],
    )
    img = img_disp = Image.open(img_path).convert("RGB")
    img = img.resize((model_input_size, model_input_size))
    img = torch.tensor(img).permute(2, 0, 1)
    img = normalize(img).float()
    return img.to(device), img_disp


def visualization(img_path, model, viz_result_path, model_input_size, device):
    img_tensor, img_disp = load_img(img_path, model_input_size, device)

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(img_disp)

    plt.subplot(122)
    plt.title("Image with Facial Keypoints")
    plt.imshow(img_disp)

    # img, _ = test_data[img_index]
    kp_s = model(img_tensor[None]).flatten().detach().cpu()
    plt.scatter(kp_s[:68] * model_input_size, kp_s[68:] * model_input_size, c="y", s=2)
    plt.savefig(viz_result_path)
