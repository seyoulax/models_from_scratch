import torch
import numpy as np

from tqdm.auto import tqdm

from .ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str = "",  # Negative prompt or empty string
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    model=None,
    seed=None,
    device="cpu",
    idle_device=None,
    tokenizer=None,
):

    with torch.no_grad():
        assert 0 <= strength <= 1, ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        if model is None:
            model = {}

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = model["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77, truncation=True
            ).input_ids
            # (B, T)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (B, T) -> (B, T, C)
            cond_context = clip(cond_tokens)
            
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77, truncation=True
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (B, T) -> (B, T, C)
            uncond_context = clip(uncond_tokens)

            # (2 * B, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77, truncation=True
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            context = clip(tokens)
        
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise NotImplementedError("only ddpm sampler is supported")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image is not None:
            encoder = model["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (B, H, W, C) -> (B, C, H, W)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)

            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # 1000 .... 0 (if 1000)
        # 1000 980 960 940 .... (if 50)
        diffusion = model["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for timestep in timesteps:
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            # (B, 4, LATENT_HEIGHT, LATENTS_WIDTH)
            model_input = latents

            if do_cfg:
                # (B, 4, LATENT_HEIGHT, LATENTS_WIDTH) -> # (2 * B, 4, LATENT_HEIGHT, LATENTS_WIDTH)
                model_input = model_input.repeat(2, 1, 1, 1)

            # predicted noise by the unet
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove noise predicted by the Unet
            latents = sampler.step(timestep, latents, model_output)
#            print("latents shape", latents.shape)

        to_idle(diffusion)

        decoder = model["decoder"]
        decoder.to(device)

        images = decoder(latents)

#        print("output images shape", images.shape)

        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # (B, H, W, C)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestamp):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestamp], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    x = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
    return x
