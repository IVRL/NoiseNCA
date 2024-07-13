import os
import argparse
import yaml
from tqdm import tqdm
import torch
import utils.misc as utils
import numpy as np
import wandb

from loss import TextureLoss
from models import NCA, NoiseNCA, PENCA

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='cfg/noisenca.yml', help="configuration")


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

    image_folder = config['image_folder']
    image_paths = [f"{image_folder}/{f}" for f in os.listdir(image_folder)]

    for idx, url in enumerate(image_paths):
        if "ipynb" in url:
            continue

        style_img = utils.imread(url, max_size=128)
        target_image = torch.from_numpy(style_img).to(device)
        target_image = target_image.permute(2, 0, 1)[None, ...]
        texture_name = url.split("/")[-1].split(".")[0]

        model_path = os.path.join(config['experiment_path'], f"{texture_name}")

        if wandb_enabled:
            wandb.login(key=config['wandb']['key'], relogin=False)
            wandb.init(project=config['wandb']['project'],
                       name=config['experiment_name'] + f"-{texture_name}",
                       dir=model_path, config=config)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        elif os.path.exists(os.path.join(model_path, "weights.pt")):
            print(f"A trained model for {texture_name} exists.")
            continue

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

        loss_log = {'total': [], 'overflow': [], 'texture': []}
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

            loss_log['total'].append(loss.item())
            loss_log['overflow'].append(overflow_loss.item())
            loss_log['texture'].append(texture_loss.item())
            if wandb_enabled:
                wandb.log(loss_log, step=epoch)

                if (epoch + 1) % 10 == 0:
                    imgs = nca.to_rgb(x[:4]).permute([0, 2, 3, 1]).detach().cpu().numpy()
                    imgs = np.hstack((np.clip(imgs, 0, 1) * 255.0).astype(np.uint8))
                    wandb.log({'NCA Output': wandb.Image(imgs, caption='NCA Output')}, step=epoch)

        # utils.statwrite(a={'loss_log': loss_log}, f=f'{model_path}/stats.json')
        if wandb_enabled:
            print("Here")
            imgs = nca.to_rgb(x[:4]).permute([0, 2, 3, 1]).detach().cpu().numpy()
            imgs = np.hstack((np.clip(imgs, 0, 1) * 255.0).astype(np.uint8))
            wandb.log({'Last Output': wandb.Image(imgs, caption='NCA Output in the last step')})

        torch.save(nca.state_dict(), f'{model_path}/weights.pt')

        if wandb_enabled:
            wandb.finish()
        del nca
        del opt


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    exp_name = config['experiment_name']
    exp_path = f'results/{exp_name}/'
    config['experiment_path'] = exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # os.system(f'cp {args.config} {exp_path}')
    main(config)
