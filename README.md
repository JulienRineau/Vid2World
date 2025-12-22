<h1> <img src="./assets/logo.png" width="35px"> Vid2World: Crafting Video Diffusion Models to Interactive World Models </h1>

<a href="https://arxiv.org/abs/2505.14357">
    <img src="https://img.shields.io/badge/paper-arXiv%3A2505.14357-b31b1b.svg" alt="arXiv Paper"/>
  </a> 
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  </a> 
  <a href='https://knightnemo.github.io/vid2world/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
  <a href='https://huggingface.co/collections/thuml/vid2world'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Page-blue'></a>

This is the official code base for the paper [Vid2World: Crafting Video Diffusion Models to Interactive World Models](https://arxiv.org/abs/2505.14357).

Give it a star 🌟 if you find our work useful!

<div align="center">


![Banner for Vid2World](./assets/v2w_overview.png)

</div>




## 🔥 News & Updates
 - 🚩 **2025-12:** We release all model checkpoints on 🤗 [Huggingface](https://huggingface.co/collections/thuml/vid2world). 

- 🚩 **2025-12:** We release code for training, inference and evaluation.

## 📋 TL;DR
We repurpose internet-scale pretrained video diffusion models into interactive world models:
- ⚙️ Converts non-causal video diffusion backbones into autoregressive, temporally causal architectures with frame-level action conditioning.
- 🦸 Enables high-fidelity, action-conditioned video simulation and scalable world model learning across robot manipulation, 3D game simulation, and open-world navigation.

## 🚀 QuickStart
### ⚙️ Environment Setup

> [!NOTE]
> The code is tested on Ubuntu 20.04, 22.04 and AlmaLinux 9.5.

First create your conda environment:
```bash
conda create -n v2w python=3.8 -y
conda activate v2w
```
Then, install dependencies:
```bash
pip install -r requirements.txt
```
For training and evaluation:
- Download the [base video model](https://huggingface.co/Doubiiu/DynamiCrafter_512/blob/main/model.ckpt) (DynamiCrafter, 320 $\times$ 512), and save it into `checkpoints/dynamicrafter_512_v1/model.ckpt`.
- Download the [pretrained i3d model](https://www.dropbox.com/scl/fi/c5nfs6c422nlpj880jbmh/i3d_torchscript.pt?rlkey=x5xcjsrz0818i4qxyoglp5bb8&dl=1) and save it into `checkpoints/i3d/i3d_torchscript.pt`.

At this point, your `checkpoints` folder should look like this:
```bash
checkpoints
├── dynamicrafter_512_v1
│   └── model.ckpt
└── i3d
    └── i3d_torchscript.pt
```
### 🤗 Models
At the moment, we provide the following models:
| File | Domain | Weight Transfer Method | Action Guidance | Training Steps | 
| ---- | ---- | ---- | ---- | ---- | 
| [Vid2World-RT1](https://huggingface.co/thuml/Vid2World-RT1) | RT-1 | Extrapolative | ✔️ | 100k   |
| [Vid2World-CSGO](https://huggingface.co/thuml/Vid2World-CSGO) | CSGO | Extrapolative | ✔️ | 100k   |
| [Vid2World-RECON](https://huggingface.co/thuml/Vid2World-RECON) | RECON | Extrapolative | ✔️ | 100k   |
| [Vid2World-RT1-NAG](https://huggingface.co/thuml/Vid2World-RT1-NAG) | RT-1 | Extrapolative | ❌ | 30k   |
| [Vid2World-RT1-Masked-NAG](https://huggingface.co/thuml/Vid2World-RT1-Masked-NAG) | RT-1 | Masked | ❌ | 30k   |
| [Vid2World-RT1-30k](https://huggingface.co/thuml/Vid2World-RT1-30k) | RT-1 | Extrapolative | ✔️ | 30k   |
| [Vid2World-RT1-Masked](https://huggingface.co/thuml/Vid2World-RT1-Masked) | RT-1 | Masked | ✔️ | 30k   |
| [Vid2World-RT1-Shift](https://huggingface.co/thuml/Vid2World-RT1-Shift) | RT-1 | Shift | ✔️ | 30k   |

Before inference, make sure you switch the `|<your_pretrained_checkpoint>|` in the config file to the path towards your local checkpoint.

## 📸 Showcases

<table>
  <tr>
    <td align="center" width="100%">
      <b>🤖 Robot Manipulation 🦾</b>
      <video src="https://github.com/user-attachments/assets/86c50cbd-55b2-437a-bd63-c37df2c4f050" width="200" controls></video>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">
      <b>🎮 Game Simulation 🕹️</b>
      <video src="https://github.com/user-attachments/assets/32533369-a20f-4922-9f03-c42ef168e98a" width="200" controls></video>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">
      <b>🗺️ Open-World Navigation 🧭</b>
      <video src="https://github.com/user-attachments/assets/eb273a00-eb87-405b-8409-34eca48de2e7" width="200" controls></video>
    </td>
  </tr>
</table>

For more showcases, check out our [Project Page](https://knightnemo.github.io/vid2world/).

## 🤖 Vid2World for Robot Manipulation
### 1. Prepare Data & Model
#### Data
To download and preprocess the used dataset:
- Download the [RT-1 Robot Action Dataset](https://robotics-transformer1.github.io) from [OXE](https://github.com/google-deepmind/open_x_embodiment).
- Run the following command in the repo to save the processed dataset to your desired local folder.
```sh
python lvdm/data/oxe_data_converter.py --dataset_name fractal20220817_data --input_path {path to downloaded OXE} --output_path {path to stored npz}
```
#### Model
For inference, download our corresponding pretrained model from 🤗[Huggingface](https://huggingface.co/collections/thuml/vid2world), check out [QuickStart](#🚀-quickstart).

### 2. Training
To launch training with the [RT-1](https://robotics-transformer1.github.io) dataset, go to `configs/manipulation/config_rt1_train.yaml` and change the `|<your_data_dir>|` into the directory where your local data directory. To launch training on 1x4 GPU cards, use the following command:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/manipulation/config_rt1_train.yaml --train  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```
For ablation experiments, we provide the corresponding configurations in `configs/ablation`.

| File | Weight Transfer Method | Action Guidance | Model Checkpoint |
| ---- | ---- | ---- | ---- |
|`config_rt1_*_masked_nag.yaml`| Masked | ❌ | [🤗Vid2World-RT1-Masked-NAG](https://huggingface.co/thuml/Vid2World-RT1-Masked-NAG) |
|`config_rt1_*_extrp_nag.yaml`| Extrapolative | ❌ | [🤗Vid2World-RT1-NAG](https://huggingface.co/thuml/Vid2World-RT1-NAG) |
| `config_rt1_*_shift.yaml` | Shift | ✔️ | [🤗Vid2World-RT1-Shift](https://huggingface.co/thuml/Vid2World-RT1-Shift) |
| `config_rt1_*_masked.yaml` | Masked | ✔️ | [🤗Vid2World-RT1-Masked](https://huggingface.co/thuml/Vid2World-RT1-Masked) |
|`config_rt1_*_all.yaml`| Extrapolative | ✔️ | [🤗Vid2World-RT1-30k](https://huggingface.co/thuml/Vid2World-RT1-30k) |
### 3. Inference
Here we provide two setups, one is generating the sequence frame by frame, which is referred to as **Auto-Regressive Generation**, and one that generates the full sequence all in one go, which we refer to as **Non-Auto-Regressive Generation**. 

Before running the experiments, make sure you download/train the corresponding checkpoints, as well as change the data paths in the config file used.

#### Auto-Regressive Generation
For auto-regressive generation, run:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base code_release_configs/manipulation/config_rt1_test_ar.yaml --val  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```
While doing ablation, switch the configuration file to the corresponding file.
#### Non-Auto-Regressive Generation
For non-auto-regressive generation, run:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base code_release_configs/manipulation/config_rt1_test_nar.yaml --val  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```

#### RT-1 Action Control Test
Test model's ability to respond to different world_vector actions (X+, X-, Y+, Y-, Z+, Z-).

First, update the config file `configs/manipulation/config_rt1_action_control_test.yaml`:
- Set `pretrained_checkpoint` to your checkpoint path
- Set `data_dir` to your RT-1 data directory

Then run:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/manipulation/config_rt1_action_control_test.yaml --val --name rt1_action_control_test --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```

Results will be saved to the directory specified in the config file's `save_dir` parameter. Each batch visualizes 8 action variants side-by-side for comparison.

## 🕹️ Vid2World for Game Simulation
### 1. Prepare Data & Model
#### Data
To download and preprocess data, please follow the steps from [DIAMOND](https://github.com/eloialonso/diamond/tree/csgo), specifically:
- Download the `.tar` files in the `dataset_dm_scraped_dust2_tars` from [this dataset repo](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning).
- Use the [provided script](https://github.com/eloialonso/diamond/blob/csgo/src/process_csgo_tar_files.py) to process the dataset for full and low res. For our purpose, we use only the `full_res` folder. 

#### Model
For inference, download our corresponding pretrained model from 🤗[Huggingface](#🤗-models), check out [QuickStart](#🚀-quickstart).

### 2. Training
To launch training with the [csgo](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) dataset, go to `configs/game/config_csgo_train.yaml` and change the `|<your_data_dir>|` into the directory where your local data directory. To launch training on 1x4 GPU cards, use the following command:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/game/config_csgo_train.yaml --train  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```
### 3. Inference

#### Standard Inference
For inference, run:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/game/config_csgo_test.yaml --val  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```

#### Long Rollout Inference on CSGO
For long rollout inference on CSGO, run:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/game/config_csgo_test_long_rollout.yaml --val  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```

#### Long Rollout Inference on OOD Games
For long rollout inference on previously unseen games (Valorant, Delta Force), run:

**Valorant:**
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/game/config_csgo_test_long_rollout_valorant.yaml --val  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 2 lightning.trainer.num_nodes=1
```

**Delta Force:**
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12879 --node_rank=0 ./main/trainer.py --base configs/game/config_csgo_test_long_rollout_delta_force.yaml --val  --name training_512_v1.0 --logdir |<your_log_dir>| --devices 2 lightning.trainer.num_nodes=1
```

## 🗺️ Vid2World for Open-World Navigation
### 1. Prepare Data & Model
#### Data
To download and preprocess data, please follow the steps from [NoMaD](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling), specifically:
- Download the [RECON](https://sites.google.com/view/recon-robot/dataset) dataset.
- Change the [ preprocessing resolution](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/data/data_utils.py#L13) to (640,480).
- Run `process_recon.py` to save the processed dataset to your desired local folder.
#### Model
For inference, download our corresponding pretrained model from 🤗[Huggingface](#🤗-models), check out [QuickStart](#🚀-quickstart).

### 2. Training
To launch training with the [RECON](https://sites.google.com/view/recon-robot/dataset) dataset, go to `configs/navigation/config_recon_train.yaml` and change the `|<your_data_dir>|` into the directory where your local data directory. To launch training on 1x4 GPU cards, use the following command:
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/navigation/config_recon_train.yaml --train --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```
### 3. Inference

Following [NWM](https://github.com/facebookresearch/nwm), we evaluate our performance under two setups: single-step generation and auto-regressive generation. While in both setups, our model is doing auto-regressive generation, the data split is different, we support both setups.

#### Single-Step Generation

Change the `|<data_dir>|` and `|<path_to_pretrained_checkpoint>|` in `configs/navigation/config_recon_test_single_step.yaml`.
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/navigation/config_recon_test_single_step.yaml --val --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```

#### Auto-Regressive Generation

Change the `|<data_dir>|` and `|<path_to_pretrained_checkpoint>|` in `configs/navigation/config_recon_test_rollout.yaml`.
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12869 --node_rank=0 ./main/trainer.py --base configs/navigation/config_recon_test_rollout.yaml --val --name training_512_v1.0 --logdir |<your_log_dir>| --devices 4 lightning.trainer.num_nodes=1
```
## 🧪 Evaluation

> [!NOTE]
> Check out [this issue](https://github.com/ssundaram21/dreamsim/issues/28) if you encounter the following error message:
> `ImportError: cannot import name 'trunc_normal_' from 'utils' (unknown location)`

For evaluation, after running the inference code, calculate the metrics by running:
```bash
python eval.py --exp_folder |<your_log_image_dir>| --env  |<rt1/csgo/recon_time/recon_rollout>|
```

## 📜 Citation

If you find our code useful, please consider citing our paper:

```bibtex
@article{huang2025vid2world0,
  title={Vid2World: Crafting Video Diffusion Models to Interactive World Models}, 
    author={Siqiao Huang and Jialong Wu and Qixing Zhou and Shangchen Miao and Mingsheng Long},
    year={2025},
  journal= {arXiv preprint arXiv:2505.14357}
}
```
## 📬 Contact
If you have any questions, please contact [huang-sq23@mails.tsinghua.edu.cn](mailto:huang-sq23@mails.tsinghua.edu.cn).

## 💡 Acknowledgement

We sincerely appreciate the following github repos for their valuable codebase we build upon:
- https://github.com/Doubiiu/DynamiCrafter 
- https://github.com/thuml/iVideoGPT
- https://github.com/facebookresearch/nwm
- https://github.com/eloialonso/diamond
- https://github.com/universome/stylegan-v