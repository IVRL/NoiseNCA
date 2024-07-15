# NoiseNCA (Artificial Life 2024) [[Project Page](https://noisenca.github.io/)]

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2404.06279)

[//]: # ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;&#41;)

![teaser](data/teaser.png)

Official implementation of the paper titled "NoiseNCA: Noisy Seed Improves Spatio-Temporal Continuity of Neural Cellular
Automata"

## Getting Started

For a quick and hands-on experimentation we suggest trying our Google Colab notebook (click on Open in
Colab button at the top of the readme).
Otherwise, you can follow the instructions below to set up the environment on your local machine.

### Installing the packages

1. Clone the repository

```bash
git clone https://github.com/IVRL/NoiseNCA.git
cd NoiseNCA
```

2. Create a new virtual environment and install the required python packages.
   The requirements are very light and are listed in the `requirements.txt` file. We use PyTorch for the implementation
   and Weights & Biases (W&B) for logging.

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

### Downloading the data

We used 45 textures from [DyNCA](https://dynca.github.io/) to train and test our model.
You can download the data by running the following command:

```bash
python3 download_textures.py
```

The data will be downloaded to the `data` directory.

## How to run

To run the code you will need to have a yaml config file that specifies the training configuration.
You can find the sample config files for Noise-NCA, Vanilla-NCA, and PE-NCA in the `configs` directory.

Running the `train.py` code will train the specified NCA model on all the textures in a given directory.
For example, to train the Noise-NCA model on the textures in the data directory, you can run the following command:

```bash
python3 train.py --config configs/Noise-NCA.yaml --data_dir data/textures/
```

### Config files

Each setting in the config file is explained in the comments.
Currently, we only support the image-guided training scheme and provide two sample config files for training with PBR
textures and single RGB textures.
The configuration and the training code for text-guided and motion-guided training schemes will be released soon.

* `configs/pbr_texture.yaml`: Contains the training settings for PBR textures.
  You can use this config file when you want MeshNCA to simultaneously learn and synthesize multiple related textures
  such as the albedo, normal, roughness, height, and ambient occlusion maps.
* `configs/single_texture.yaml`: Contains the training settings for RGB textures.

#### Example:

The loss_type key in the config file specifies the loss function used for training the model.
The supported values are `OT`, `Gram`, and 'SlW' for optimal transport, Gram matrix, and sliced Wasserstein loss,
respectively. In the paper we used the OT loss.

```yaml
loss:
  attr:
    loss_type: "OT"
```

For the Noise-NCA model, you can specify the noise level for each target texture. In the sample
config file, we provide the noise level for each texture that was used in the paper.

```yaml
model:
  type: "NoiseNCA"
  noise_levels: { default: 0.10, # This is the default noise level for images that are not listed below
                  bumpy_0081: 0.25,
                  bumpy_0169: 0.25,
  } # We found these noise levels to work well for the given textures.
  attr:
    chn: 12
    fc_dim: 96
```

To record the training logs in wandb (Weights and Biases), you can set the `wandb` key in the config file.

```yaml
  wandb:
    project: "Project Name"
    key: "Your API Key"
```

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
