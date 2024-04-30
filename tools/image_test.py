"""Testing diffusion kernel regression on image dataset."""

from PIL import Image
from diffusion_kernel_regression import DiffusionKernelRegressionImage, ImageDataset


if __name__ == "__main__":
    batch_size = 1
    check_knn = True
    diffusion_steps = 1000
    scheduler_type = "squaredcos_cap_v2"
    image_path = "/home/harvey/Data/celeba_hq_256"
    dataset = ImageDataset(image_path)

    generator = DiffusionKernelRegressionImage(latents=dataset.latents, vae=dataset.vae, image_processor=dataset.image_processor, knn_max=100, diffusion_steps=diffusion_steps, scheduler_type=scheduler_type)
    image_list = generator.sample_image(batch_size=batch_size, check_knn=check_knn)
    for image in image_list:
        image.show()
