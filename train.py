import os
from utils.affiliated_utils import backup_empty_files, load_param, copy_net
from utils import init_env

import argparse

import torch
from utils.collate_utils import collate, SampleDataset
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(0)


def get_loader(cfg):
    train_dataset = instantiate_from_config(cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)

    test_dataset = instantiate_from_config(cfg.test_dataset.USOD10K)
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.USOD10K), interval=10)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.USOD), interval=30)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return train_loader, test_loader


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['WANDB_API_KEY'] = '*'

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--load_pretrained', type=str, default=None)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--results_folder', type=str, default='results/',
    help='None for saving in wandb folder.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_min', type=float, default=8e-6)
    parser.add_argument('--project_name', type=str, default='USOD')

    cfg = add_args(parser)

    copy_net(cfg)

    config_pretty(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)  # model.net.net


    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)
    # load_param('results/net_asym_modify_6_x_f_add/model-best.pt', diffusion_model, mode='model')

    if cfg.load_pretrained not in [None, '']:
        load_param(cfg.load_pretrained, diffusion_model, mode='model')

    train_loader, test_loader = get_loader(cfg)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epoch, eta_min=cfg.lr_min)

    trainer = Trainer(
        diffusion_model, train_loader, test_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None if cfg.num_workers == 0 else 'wandb',  # debug
        cfg=cfg,
    )
    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    trainer.train()
    # backup_empty_files()
