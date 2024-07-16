import os
import shutil
import argparse
import yaml
from tqdm import tqdm
import torch
import utils.misc as utils
import numpy as np
import wandb
from PIL import Image

from loss import TextureLoss
from models import NCA, NoiseNCA, PENCA

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/Noise-NCA.yml', help="Path to the config file")
parser.add_argument('--data_dir', type=str, default='data/textures/', help="Texture images directory")

os.environ["WANDB_SILENT"] = "true"


def get_nca_model(config, texture_name):
    model_type = config['model']['type']
    if model_type == 'NCA':
        return NCA(**config['model']['attr'])
    elif model_type == 'NoiseNCA':
        noise_levels = config['model']['noise_levels']
        if texture_name in noise_levels:
            noise_level = noise_levels[texture_name]
        else:
            noise_level = noise_levels['default']
        return NoiseNCA(noise_level=noise_level, **config['model']['attr'])
    elif model_type == 'PENCA':
        return PENCA(**config['model']['attr'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(config):
    wandb_enabled = 'wandb' in config
    if wandb_enabled:
        wandb.login(key=config['wandb']['key'], relogin=True)

    device = torch.device(config['device'])
    config['loss']['attr']['device'] = device
    config['model']['attr']['device'] = device
    loss_fn = TextureLoss(**config['loss']['attr']).to(device)

    data_dir = config['data_dir']
    image_paths = [f"{data_dir}/{f}" for f in os.listdir(data_dir)]

    for idx, url in enumerate(image_paths):
        if "ipynb" in url:
            continue

        style_img = utils.imread(url, max_size=128)
        target_image = torch.from_numpy(style_img).to(device)
        target_image = target_image.permute(2, 0, 1)[None, ...]
        texture_name = url.split("/")[-1].split(".")[0]

        model_path = os.path.join(config['experiment_path'], f"{texture_name}")
        log_path = os.path.join(model_path, "logs")
        if not os.path.exists(model_path):
            os.makedirs(log_path)
        elif os.path.exists(os.path.join(model_path, "weights.pt")):
            print(f"A trained model for {texture_name} exists.")
            continue
        else:
            shutil.rmtree(model_path)
            os.makedirs(log_path)

        if wandb_enabled:
            name = config['experiment_name'] + f"-{texture_name}"
            wandb_run = wandb.init(project=config['wandb']['project'],
                                   name=name, dir=log_path, config=config)

        nca = get_nca_model(config, texture_name).to(device)

        opt = torch.optim.Adam(nca.parameters(), config['training']['lr'], capturable=True)

        lr_sched = None
        if 'type' not in config['training']['scheduler'] or config['training']['scheduler']['type'] == 'MultiStep':
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, **config['training']['scheduler']['attr'])
        elif config['training']['scheduler']['type'] == 'Cyclic':
            lr_sched = torch.optim.lr_scheduler.CyclicLR(opt, **config['training']['scheduler']['attr'])

        batch_size = config['training']['batch_size']
        iterations = config['training']['iterations']
        alpha = config['training']['overflow_weight']

        step_range = config['training']['nca']['step_range']
        inject_seed_step = config['training']['nca']['inject_seed_step']
        pool_size = config['training']['nca']['pool_size']

        with torch.no_grad():
            pool = nca.seed(pool_size).to(device)

        pbar = tqdm(range(iterations), desc=f"Training {idx + 1}/{len(image_paths)} on {texture_name}")
        for epoch in pbar:
            with torch.no_grad():
                batch_idx = np.random.choice(pool_size, batch_size, replace=False)
                x = pool[batch_idx]
                if epoch % inject_seed_step == 0:
                    x[:1] = nca.seed(1)

            step_n = np.random.randint(step_range[0], step_range[1])

            for _ in range(step_n):
                x = nca(x)

            overflow_loss = (x - x.clamp(-1.0, 1.0)).abs().sum()
            texture_loss, texture_loss_per_img = loss_fn(target_image, nca.to_rgb(x))
            loss = texture_loss + alpha * overflow_loss

            with torch.no_grad():
                loss.backward()
                for p in nca.parameters():
                    p.grad /= (p.grad.norm() + 1e-8)  # normalize gradients
                opt.step()
                opt.zero_grad()
                lr_sched.step()

                pool[batch_idx] = x

            if (epoch + 1) % config['training']['log_interval'] == 0:
                imgs = nca.to_rgb(x[:4]).permute([0, 2, 3, 1]).detach().cpu().numpy()
                imgs = np.hstack((np.clip(imgs, 0, 1) * 255.0).astype(np.uint8))
                if wandb_enabled:
                    wandb_run.log({'NCA Output': wandb.Image(imgs, caption='NCA Output')}, step=epoch)
                else:
                    Image.fromarray(imgs).save(f'{log_path}/epoch-{epoch}.png')

            if wandb_enabled:
                loss_log = {
                    'total': loss.item(),
                    'overflow': overflow_loss.item(),
                    'texture': texture_loss.item()
                }
                wandb_run.log(loss_log, step=epoch)

        torch.save(nca.state_dict(), f'{model_path}/weights.pt')

        if wandb_enabled:
            wandb_run.finish()
        del nca
        del opt


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config['data_dir'] = args.data_dir
    exp_name = config['experiment_name']
    exp_path = f'results_new/{exp_name}/'
    config['experiment_path'] = exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    main(config)
