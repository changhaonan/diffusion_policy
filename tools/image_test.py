"""Testing diffusion kernel regression on image dataset."""
from diffusion_kernel_regression import DiffusionKernelRegression, ImageDataset


if __name__ == "__main__":
    image_path = 'data/celeba_64x64.npz'
    dataset = ImageDataset(image_path)