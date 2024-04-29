"""Testing diffusion kernel regression on image dataset."""
from diffusers import AutoencoderKL
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import torch


if __name__ == "__main__":
    ## Load VAE
    torch.set_grad_enabled(False)
    dev = 0
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    )
    vae = vae.to(dev)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    
    img = Image.open(f'./arm/0.png')

    ## Encode
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    pixel_values = image_processor.numpy_to_pt(image_processor.normalize(
        image_processor.resize(
        image_processor.pil_to_numpy(img), 255, 255)))
    latents = vae.encode(pixel_values.to(dev)).latent_dist.sample()

    print(latents.shape)

    ## Decode
    scaling_factor = 1
    decode_pixel_values = vae.decode(latents / scaling_factor, return_dict=False)[0]
    decode_img = image_processor.postprocess(decode_pixel_values.detach(), do_denormalize=[True])[0]