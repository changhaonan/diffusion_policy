"""Testing diffusion kernel regression on image dataset."""

from PIL import Image
from diffusion_kernel_regression import DiffusionKernelRegressionImage, ImageDataset


if __name__ == "__main__":
    image_path = "/home/harvey/Data/celeba_hq_256"
    dataset = ImageDataset(image_path)

    generator = DiffusionKernelRegressionImage(latents=dataset.latents, vae=dataset.vae, image_processor=dataset.image_processor, knn_max=100, diffusion_steps=100, scheduler_type="linear")
    image = generator.sample_image(batch_size=1)
    image.show()
