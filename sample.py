import sys

import torch
from utils.train_utils import set_random_seed

import os
import argparse
from pathlib import Path

from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(0)


def get_loader(cfg):
    usod_test_dataset = instantiate_from_config(cfg.test_dataset.USOD)
    usod10k_test_dataset = instantiate_from_config(cfg.test_dataset.USOD10K)

    usod_test_loader = DataLoader(
        usod_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    usod10k_test_loader = DataLoader(
        usod10k_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )

    return usod10k_test_loader, usod_test_loader


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./results/model-best.pt')
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--results_folder', type=str, default='./results/maps/')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=5)
    # parser.add_argument('--target_dataset', nargs='+', type=str, default=['USOD10K','USOD'])
    parser.add_argument('--target_dataset', nargs='+', type=str, default=['USOD10K'])
    # parser.add_argument('--target_dataset', nargs='+', type=str, default=['USOD'])
    parser.add_argument('--time_ensemble', type=bool, default=True)
    parser.add_argument('--batch_ensemble', type=bool, default=False)

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    """
        Hack config here.
    """
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    usod10k_test_loader, usod_test_loader = get_loader(cfg)

    _ = instantiate_from_config(cfg.cond_uvit,
                                conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=_)

    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    usod10k_test_loader, usod_test_loader = \
        trainer.accelerator.prepare(usod10k_test_loader, usod_test_loader)

    dataset_map = {
        'USOD10K': usod10k_test_loader,
        'USOD': usod_test_loader
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        mask_path = Path(cfg.test_dataset.USOD10K.params.image_root).parent.parent.parent
        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)
        if cfg.batch_ensemble:
            mae, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            mae, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            mae, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')

        # if trainer.accelerator.is_main_process:
        #     from utils.eval import eval
        #
        #     eval_score = eval(
        #         mask_path=mask_path,
        #         pred_path=cfg.results_folder,
        #         dataset_name=dataset_name)
        trainer.accelerator.wait_for_everyone()
