#!/usr/bin/env python
"""finetune_fold.py - fine-tune one pretrained region model on one fold.

Generalized from the research code (``scripts/09_finetune_one_fold.py``), which
is what produced the fine-tuned models in the paper. Use this when the target
cohort's folds are defined externally (e.g. a stratified, subject-level split)
rather than by the internal KFold of ``train_cv.py``.

Behaviour preserved from the original:
  * the pretrained checkpoint of the SAME region and the SAME fold is loaded
    (``cv-{fold}-{load_epoch}.pth.tar``, falling back to ``cv-{fold}.pth.tar``)
  * NO layer is frozen — every parameter is fine-tuned
  * the optimizer (and scheduler) are rebuilt after loading, so no pretraining
    optimizer state carries over
  * MSE drives the updates, MAE is the model-selection metric
  * the epoch with the lowest validation MAE becomes ``cv-{fold}-best.pth.tar``

``--from_scratch`` reproduces the paper's from-scratch baseline: identical
recipe, random (Kaiming) initialisation, no pretrained weights.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regionbae import (  # noqa: E402
    CNN, CNN_Trainer, RegionConfig, Region_Dataset,
    get_logger, initialize_weights, make_scheduler, seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv', required=True)
    p.add_argument('--region', required=True)
    p.add_argument('--region_config', required=True)
    p.add_argument('--atlas', default=None)
    p.add_argument('--dilate_radius', type=int, default=None)
    p.add_argument('--fold', type=int, required=True,
                   help='fold index; selects the pretrained checkpoint and names the outputs')

    p.add_argument('--model_load_folder', default='',
                   help='directory with the pretrained cv-{fold}[-{epoch}].pth.tar')
    p.add_argument('--load_epoch', default='40')
    p.add_argument('--from_scratch', action='store_true',
                   help='random init instead of loading pretrained weights')

    p.add_argument('--model_dir', required=True)
    p.add_argument('--results_dir', required=True)

    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=5e-5)
    p.add_argument('--lr_scheduler_choice', type=int, default=0, choices=[0, 1])
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--gpu', type=int, default=1)

    p.add_argument('--image_col', default='imgs')
    p.add_argument('--age_col', default='age')
    p.add_argument('--id_col', default='subjectID')

    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='RegionBAE')
    p.add_argument('--run_tag', default='finetune', help='label used in the W&B run name')
    return p.parse_args()


def main():
    args = parse_args()
    gpu = bool(args.gpu) and torch.cuda.is_available()
    device = torch.device('cuda' if gpu else 'cpu')
    seed_everything(args.seed)

    region_config = RegionConfig.from_json(
        args.region_config, atlas_path=args.atlas, dilate_radius=args.dilate_radius)
    labels = region_config.labels_for(args.region)

    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    print('=' * 60)
    print(f'[finetune_fold] region={args.region} labels={labels} fold={args.fold}')
    print(f'  train / val    : {len(train_df)} / {len(val_df)}')
    print(f'  atlas          : {region_config.atlas_path} (dilate {region_config.dilate_radius})')
    print(f'  epochs={args.epochs} batch={args.batch_size} lr={args.lr} '
          f'wd={args.weight_decay} sched={args.lr_scheduler_choice} patience={args.patience}')
    print(f'  init           : {"random (from scratch)" if args.from_scratch else args.model_load_folder}')
    print(f'  device         : {device}')
    print('=' * 60)

    def build(df):
        return Region_Dataset(
            df, None, args.region, region_config=region_config,
            image_col=args.image_col, age_col=args.age_col, id_col=args.id_col)

    train_ds, val_ds = build(train_df), build(val_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=RandomSampler(train_ds),
                              collate_fn=train_ds.collate_fn, pin_memory=gpu,
                              num_workers=args.n_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=SequentialSampler(val_ds),
                            collate_fn=val_ds.collate_fn, pin_memory=gpu,
                            num_workers=args.n_workers)
    valid_ids = val_df[args.id_col].tolist() if args.id_col in val_df.columns else None

    model = CNN(in_channels=1)
    if args.from_scratch:
        model.apply(initialize_weights)
        print('[FROM_SCRATCH] applied initialize_weights (Kaiming); skipping pretrained load')
    model = model.to(device)

    logger = get_logger(
        use_wandb=args.wandb, project=args.wandb_project,
        name=f'{args.region}_fold{args.fold}_{args.run_tag}',
        group=f'{args.region}_{args.run_tag}', job_type=f'fold{args.fold}', reinit=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, len(train_loader), args.lr_scheduler_choice, args.lr)

    trainer = CNN_Trainer(
        model=model,
        model_load_folder=args.model_load_folder,
        model_save_folder=str(model_dir),
        results_folder=str(results_dir),
        dataloader_train=train_loader,
        dataloader_valid=val_loader,
        dataloader_test=None,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        cv_num=args.fold,
        region=args.region,
        valid_ids=valid_ids,
        device=device,
        patience=args.patience,
        logger=logger,
    )

    if not args.from_scratch:
        if not args.model_load_folder:
            raise ValueError('--model_load_folder is required unless --from_scratch is given')
        load_epoch = args.load_epoch if args.load_epoch == 'best' else int(args.load_epoch)
        trainer.load(args.fold, load_epoch, gpu)

    # Fresh optimizer + scheduler after loading pretrained weights.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer.optimizer = optimizer
    trainer.scheduler = make_scheduler(optimizer, len(train_loader), args.lr_scheduler_choice, args.lr)

    trainer.train()

    if trainer.valid_mae_list:
        best_epoch = int(np.argmin(trainer.valid_mae_list)) + 1
        best_src = model_dir / f'cv-{args.fold}-{best_epoch}.pth.tar'
        best_dst = model_dir / f'cv-{args.fold}-best.pth.tar'
        if best_src.exists():
            shutil.copy2(best_src, best_dst)
            print(f'[BEST] epoch {best_epoch} '
                  f'(val MAE={min(trainer.valid_mae_list):.4f}) -> {best_dst.name}')
        else:
            print(f'[WARN] best checkpoint not found at {best_src}')

    logger.finish()
    print('[DONE]')


if __name__ == '__main__':
    main()
