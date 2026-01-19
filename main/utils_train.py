import os, re
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch

# Try to import wandb - optional dependency
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    mainlogger.warning("wandb not installed. Run 'pip install wandb' for experiment tracking.")
from collections import OrderedDict
from lvdm.basics import CausalConv1d, CausalConv2d, CausalConv3d
import numpy as np


def get_conv_type(model, key):
    """
    get the specific dimension (1D, 2D, 3D) of conv_nd by model structure
    """
    module = model
    sub_keys = key.split(".")
    
    # get module layer by layer
    for sub_key in sub_keys:
        if hasattr(module, sub_key):
            module = getattr(module, sub_key)
        else:
            return None  # key doesn't match model, maybe extra parameters

    # check module type
    if isinstance(module, CausalConv1d):
        return "conv1d"
    elif isinstance(module, CausalConv2d):
        return "conv2d"
    elif isinstance(module, CausalConv3d):
        return "conv3d"
    else:
        return None  # other types of layers, no modification

def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir, on_debug, name=None, rank=0):
    """
    Get trainer logger configuration(s).

    Returns tensorboard config by default, plus wandb config if enabled.

    Args:
        lightning_config: Lightning configuration
        logdir: Log directory path
        on_debug: Debug mode flag
        name: Experiment name for wandb
        rank: Process rank for distributed training

    Returns:
        List of logger configs or single logger config
    """
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

    # Check if wandb is enabled
    wandb_cfg = get_wandb_logger_config(lightning_config, logdir, name, rank)
    if wandb_cfg is not None:
        return [logger_cfg, wandb_cfg]

    return logger_cfg


def get_wandb_logger_config(lightning_config, logdir, name=None, rank=0):
    """
    Get wandb logger configuration if enabled.

    Args:
        lightning_config: Lightning configuration with optional 'wandb' section
        logdir: Log directory path
        name: Experiment name
        rank: Process rank for distributed training

    Returns:
        WandbLogger config dict or None if disabled/unavailable
    """
    # Check if wandb is available
    if not WANDB_AVAILABLE:
        return None

    # Check if wandb is enabled in config
    wandb_config = check_config_attribute(lightning_config, "wandb")
    if wandb_config is None:
        return None

    enabled = check_config_attribute(wandb_config, "enabled")
    if not enabled:
        mainlogger.info("Wandb logging disabled in config")
        return None

    # Get wandb configuration parameters
    project = check_config_attribute(wandb_config, "project") or "vid2world"
    entity = check_config_attribute(wandb_config, "entity")  # Can be None for personal workspace
    tags = check_config_attribute(wandb_config, "tags") or []
    notes = check_config_attribute(wandb_config, "notes") or ""
    resume = check_config_attribute(wandb_config, "resume") or "allow"
    log_model = check_config_attribute(wandb_config, "log_model") or False
    save_code = check_config_attribute(wandb_config, "save_code") or True

    # Convert OmegaConf list to Python list if needed
    if hasattr(tags, '__iter__') and not isinstance(tags, (list, str)):
        tags = list(tags)

    # Only initialize on rank 0
    if rank != 0:
        mainlogger.info(f"Skipping wandb init on rank {rank}")
        return None

    mainlogger.info(f"Initializing wandb logger: project={project}, name={name}")

    wandb_logger_cfg = {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "project": project,
            "name": name,
            "save_dir": logdir,
            "tags": tags,
            "notes": notes,
            "resume": resume,
            "log_model": log_model,
        }
    }

    if entity is not None:
        wandb_logger_cfg["params"]["entity"] = entity

    return wandb_logger_cfg


def init_wandb(lightning_config, model_config, name, logdir, rank=0):
    """
    Initialize wandb run on rank 0 only.

    This should be called before trainer initialization to ensure
    hyperparameters are logged correctly.

    Args:
        lightning_config: Lightning configuration
        model_config: Model configuration (for hyperparameter logging)
        name: Experiment name
        logdir: Log directory
        rank: Process rank

    Returns:
        wandb run object or None
    """
    if not WANDB_AVAILABLE:
        return None

    wandb_config = check_config_attribute(lightning_config, "wandb")
    if wandb_config is None or not check_config_attribute(wandb_config, "enabled"):
        return None

    if rank != 0:
        return None

    project = check_config_attribute(wandb_config, "project") or "vid2world"
    entity = check_config_attribute(wandb_config, "entity")
    tags = check_config_attribute(wandb_config, "tags") or []
    notes = check_config_attribute(wandb_config, "notes") or ""
    resume = check_config_attribute(wandb_config, "resume") or "allow"
    save_code = check_config_attribute(wandb_config, "save_code") or True

    # Convert OmegaConf to dict for wandb config
    config_dict = {
        "model": OmegaConf.to_container(model_config, resolve=True),
        "lightning": OmegaConf.to_container(lightning_config, resolve=True),
    }

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        dir=logdir,
        tags=list(tags) if not isinstance(tags, list) else tags,
        notes=notes,
        resume=resume,
        save_code=save_code,
        config=config_dict,
    )

    mainlogger.info(f"Wandb initialized: {run.url}")
    return run

def get_trainer_strategy(lightning_config):
    """Get trainer strategy configuration for Lightning 2.x.

    Returns 'ddp' string by default for standard DDP training.
    DDPShardedStrategy was removed in Lightning 2.x.
    """
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        # Default to simple DDP strategy (works with Lightning 2.x)
        return "ddp"

def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if 'state_dict' in pl_sd.keys():
                model.load_state_dict(pl_sd["state_dict"], strict=True)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                model.load_state_dict(new_pl_sd, strict=True)
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")

    return model

def load_checkpoints_causal(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), f"Error: Pre-trained checkpoint NOT found at: {pretrained_ckpt}"
        mainlogger.info(">>> Load weights from pretrained checkpoint")
        
        # Validate and set weight transfer mode
        weight_transfer_mode = getattr(model_cfg, 'better_weight_transfer', None)
        if weight_transfer_mode and weight_transfer_mode not in ['masked', 'extrapolative', 'shift']:
            raise ValueError(f"Invalid better_weight_transfer mode: {weight_transfer_mode}. Must be 'masked' or 'extrapolative' or 'shift'")
        
        mainlogger.info(f"Causal Better weight transfer mode: {weight_transfer_mode}")
        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if 'state_dict' in pl_sd.keys():
                state_dict = pl_sd["state_dict"]
                try:
                    model.load_state_dict(state_dict, strict=True)
                except:
                    new_pl_sd = OrderedDict()
                    renamed_count = 0
                    for k, v in state_dict.items():
                        if "temopral_conv" in k and "conv" in k and any(x in k for x in ["1", "2", "3", "4"]):
                            # detect convolution layer type and rename
                            if k.endswith(".weight") or k.endswith(".bias"):
                                conv_type = get_conv_type(model, k.replace(".weight", "").replace(".bias", ""))
                                if conv_type:
                                    if weight_transfer_mode == "masked":
                                        v = masked_weight_transfer(k, conv_type, v)
                                    elif weight_transfer_mode == "extrapolative":
                                        v = extrapolative_weight_transfer(k, conv_type, v)
                                    # for shift mode, we do not need to additionally compute any values
                                    # create new key with correct convolution type
                                    new_key = k.replace(".weight", f".{conv_type}.weight").replace(".bias", f".{conv_type}.bias")
                                    new_pl_sd[new_key] = v
                                    renamed_count += 1
                                    continue
                        # handle output layer keys
                        if "model.diffusion_model.out.2.weight" in k:
                            if weight_transfer_mode == "masked":
                                v = masked_weight_transfer(k, "conv2d", v)
                            elif weight_transfer_mode == "extrapolative":
                                v = extrapolative_weight_transfer(k, "conv2d", v)
                            new_key = k.replace("weight", "conv2d.weight")
                            new_pl_sd[new_key] = v
                            renamed_count += 1
                            continue
                        if "model.diffusion_model.out.2.bias" in k:
                            new_key = k.replace("bias", "conv2d.bias")
                            new_pl_sd[new_key] = v
                            renamed_count += 1
                            continue
                        new_pl_sd[k] = v
                    if renamed_count > 0:
                        mainlogger.info(f"Renamed {renamed_count} checkpoint keys for causal attention compatibility")
                    try:
                        model.load_state_dict(new_pl_sd, strict=True)
                    except:
                        missing_keys, unexpected_keys = model.load_state_dict(new_pl_sd, strict=False)
                        mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
                        mainlogger.info(f"Missing keys: {missing_keys}")
                        mainlogger.info(f"Unexpected keys: {unexpected_keys}") 
                        assert all(
                            ("action_emb_proj" in item or "action_emb_preprocess" in item)
                            for item in missing_keys
                        )
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                renamed_count = 0
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                    for k, v in new_pl_sd.items():
                        if "temopral_conv" in k and "conv" in k and any(x in k for x in ["1", "2", "3", "4"]):
                            # detect convolution layer type and rename
                            if k.endswith(".weight") or k.endswith(".bias"):
                                conv_type = get_conv_type(model, k.replace(".weight", "").replace(".bias", ""))
                                if conv_type:
                                    if weight_transfer_mode == "masked":
                                        v = masked_weight_transfer(k, conv_type, v)
                                    elif weight_transfer_mode == "extrapolative":
                                        v = extrapolative_weight_transfer(k, conv_type, v)
                                    # create new key with correct convolution type
                                    new_key = k.replace(".weight", f".{conv_type}.weight").replace(".bias", f".{conv_type}.bias")
                                    new_pl_sd[new_key] = v
                                    renamed_count += 1
                                    continue
                        # handle output layer keys
                        if "model.diffusion_model.out.2.weight" in k:
                            if weight_transfer_mode == "masked":
                                v = masked_weight_transfer(k, "conv2d", v)
                            elif weight_transfer_mode == "extrapolative":
                                v = extrapolative_weight_transfer(k, "conv2d", v)
                            new_key = k.replace("weight", "conv2d.weight")
                            new_pl_sd[new_key] = v
                            renamed_count += 1
                            continue
                        if "model.diffusion_model.out.2.bias" in k:
                            new_key = k.replace("bias", "conv2d.bias")
                            new_pl_sd[new_key] = v
                            renamed_count += 1
                            continue
                        new_pl_sd[k] = v
                if renamed_count > 0:
                    mainlogger.info(f"Renamed {renamed_count} checkpoint keys for causal attention compatibility (deepspeed)")
                try:
                    model.load_state_dict(new_pl_sd, strict=True)
                except:
                    missing_keys, unexpected_keys = model.load_state_dict(new_pl_sd, strict=False)
                    mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
                    mainlogger.info(f"Missing keys: {missing_keys}")
                    mainlogger.info(f"Unexpected keys: {unexpected_keys}")
                    assert all(
                        ("action_emb_proj" in item or "action_emb_preprocess" in item)
                        for item in missing_keys
                    )
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")

    
    mainlogger.info(">>> Model checkpoint loaded.")
    return model

def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def masked_weight_transfer(key, conv_type, value):
    # determine convnd: always in the form of conv1d, conv2d, conv3d
    N=int(conv_type[4])
    # determine whether it's weight or bias
    if key.endswith(".bias"):
        return value
    elif key.endswith(".weight"):
        # causal conv's kernel is implemented as shape: [C_1,C_2,T,(other N-1 non-causal dims)]
        kernel_t=value.shape[-N]
        assert kernel_t%2==1, "Kernel Size should all be odd in DynamiCrafter"
        if kernel_t==1:
            return value # no need to transfer any weight
        else:
            x=(kernel_t-1)//2
            new_value=torch.zeros_like(value,device=value.device)
            # keep the causal part
            # fill the non-causal part with zeros
            if N == 1:
                new_value[:, :, x:] = value[:, :, :x+1]
            elif N == 2:
                new_value[:, :, x:, :] = value[:, :, :x+1, :]
            elif N == 3:
                new_value[:, :, x:, :, :] = value[:, :, :x+1, :, :]
            else:
                raise ValueError(f"Invalid conv type: {conv_type}")
            return new_value
    else:
        raise ValueError(f"Invalid key for conv weight transfer: {key}")

def extrapolative_weight_transfer(key, conv_type, value):
    # determine convnd: always in the form of conv1d, conv2d, conv3d
    N = int(conv_type[4])
    if key.endswith(".bias"):
        return value
    elif key.endswith(".weight"):
        # causal conv's kernel is implemented as shape: [C_out, C_in, T, (other N-1 non-causal dims)]
        kernel_t = value.shape[2]  # time dimension
        assert kernel_t % 2 == 1, "Kernel Size should all be odd"
        if kernel_t == 1:
            return value
        new_value = torch.zeros_like(value, device=value.device)
        if kernel_t == 3:
            w0 = value[:, :, 0]
            w1 = value[:, :, 1]
            w2 = value[:, :, 2]
            new_value[:, :, 0] = 0
            new_value[:, :, 1] = w0 - w2
            new_value[:, :, 2] = w1 + 2 * w2
        elif kernel_t == 1:
            new_value = value
        else:
            raise NotImplementedError("Only kernel size 1 or 3 is supported currently")

        return new_value
    else:
        return value