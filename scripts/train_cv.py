#!/usr/bin/env python
"""train_cv.py - k-fold training of one region's brain-age model.

Generalized from the research code (``src_v2/main_cv.py``). One run trains ONE
region: a fresh ``CNN`` instance with its own parameters, its own checkpoint
directory and its own results directory. Run it once per region (see
``examples/run_training.sh``).

Modes:
  train      random init (Kaiming via ``initialize_weights``)
  finetune   initialise from a pretrained checkpoint of the SAME region and the
             SAME fold, then train every layer (no freezing)

Outputs (per fold F):
  {model_dir}/cv-{F}-{epoch}.pth.tar   every epoch
  {model_dir}/cv-{F}-best.pth.tar      lowest validation MAE
  {results_dir}/{F}/best/{region}.pkl  predictions of the best epoch
  {results_dir}/{region}_oof_predictions.csv   concatenated out-of-fold predictions
"""

import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regionbae import (  # noqa: E402
    CNN, CNN_Trainer, RegionConfig, Region_Dataset,
    get_logger, initialize_weights, make_scheduler, seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort_csv', required=True,
                   help="cohort table; needs columns 'imgs' (path to preprocessed "
                        "volume) and 'age'; 'subjectID' is kept in the outputs if present")
    p.add_argument('--region', required=True, help='region name as defined in --region_config')
    p.add_argument('--region_config', required=True, help='path to the region JSON config')
    p.add_argument('--atlas', default=None, help='override the atlas path in the config')
    p.add_argument('--dilate_radius', type=int, default=None,
                   help='override the dilation radius in the config (atlas voxels)')

    p.add_argument('--mode', choices=['train', 'finetune'], default='train')
    p.add_argument('--model_load_folder', default='',
                   help='finetune: directory holding the pretrained cv-{fold}[-{epoch}].pth.tar')
    p.add_argument('--load_epoch', default='40',
                   help="epoch of the pretrained checkpoint, or 'best' (finetune only)")

    p.add_argument('--model_dir', required=True, help='where checkpoints are written')
    p.add_argument('--results_dir', required=True, help='where prediction pickles are written')

    p.add_argument('--n_splits', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--start_cv', type=int, default=0, help='resume from this fold')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=5e-5)
    p.add_argument('--lr_scheduler_choice', type=int, default=0, choices=[0, 1],
                   help='0: none, 1: CustomCosineAnnealingWarmUpRestarts')
    p.add_argument('--patience', type=int, default=0,
                   help='early-stopping patience in epochs; 0 disables')
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=1)

    p.add_argument('--test_csv', default='',
                   help='optional held-out cohort scored after every epoch')
    p.add_argument('--image_col', default='imgs')
    p.add_argument('--age_col', default='age')
    p.add_argument('--id_col', default='subjectID')

    p.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
    p.add_argument('--wandb_project', default='RegionBAE')
    return p.parse_args()


def make_loader(dataset, batch_size, n_workers, gpu, shuffle):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=dataset.collate_fn, pin_memory=gpu, num_workers=n_workers,
    )


def collect_oof_predictions(results_dir, region, n_splits, out_csv):
    """Concatenate the per-fold best-epoch validation predictions into one CSV."""
    rows = []
    for fold in range(n_splits):
        pkl_path = Path(results_dir) / str(fold) / 'best' / f'{region}.pkl'
        if not pkl_path.exists():
            print(f'[WARN] missing {pkl_path}; skipping fold {fold} in the OOF table')
            continue
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        pred = np.asarray(result['pred_ages'], dtype=float)
        true = np.asarray(result['true_ages'], dtype=float)
        ids = result.get('valid_ids') or [None] * len(pred)
        for sid, t, pr in zip(ids, true, pred):
            rows.append({'subjectID': sid, 'fold': fold, 'region': region,
                         'chronological_age': t, 'predicted_age': pr,
                         'raw_bag': pr - t})
    if not rows:
        print('[WARN] no fold predictions found; OOF table not written')
        return
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'[OOF] {len(df)} rows -> {out_csv}  (MAE {np.abs(df["raw_bag"]).mean():.4f})')


def main():
    args = parse_args()
    gpu = bool(args.gpu) and torch.cuda.is_available()
    device = torch.device('cuda' if gpu else 'cpu')
    seed_everything(args.seed)

    region_config = RegionConfig.from_json(
        args.region_config, atlas_path=args.atlas, dilate_radius=args.dilate_radius)
    labels = region_config.labels_for(args.region)

    cohort_df = pd.read_csv(args.cohort_csv)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'[train_cv] region={args.region}  labels={labels}  mode={args.mode}')
    print(f'  atlas          : {region_config.atlas_path}')
    print(f'  dilate_radius  : {region_config.dilate_radius}')
    print(f'  cohort         : {args.cohort_csv}  (n={len(cohort_df)})')
    print(f'  folds          : {args.n_splits} (KFold shuffle, seed={args.seed})')
    print(f'  epochs={args.epochs} batch={args.batch_size} lr={args.lr} '
          f'wd={args.weight_decay} sched={args.lr_scheduler_choice} patience={args.patience}')
    print(f'  model_dir      : {model_dir}')
    print(f'  results_dir    : {results_dir}')
    print(f'  device         : {device}')
    print('=' * 60)

    test_df = pd.read_csv(args.test_csv) if args.test_csv else None

    kf = KFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)
    dataset_indices = list(cohort_df.index)

    for cv_num, (train_indices, valid_indices) in enumerate(kf.split(dataset_indices)):
        if cv_num < args.start_cv:
            continue
        print(f'\n<<< Fold {cv_num + 1}/{args.n_splits} >>>')

        def build(df, indices):
            return Region_Dataset(
                df, indices, args.region, region_config=region_config,
                image_col=args.image_col, age_col=args.age_col, id_col=args.id_col)

        train_dataset = build(cohort_df, train_indices)
        valid_dataset = build(cohort_df, valid_indices)
        print(f'Train / valid size: {len(train_dataset)} / {len(valid_dataset)}')

        dataloader_train = make_loader(train_dataset, args.batch_size, args.n_workers, gpu, True)
        dataloader_valid = make_loader(valid_dataset, args.batch_size, args.n_workers, gpu, False)
        dataloader_test = None
        if test_df is not None:
            test_dataset = build(test_df, list(test_df.index))
            dataloader_test = make_loader(test_dataset, args.batch_size, args.n_workers, gpu, False)

        valid_ids = (cohort_df.loc[valid_indices, args.id_col].tolist()
                     if args.id_col in cohort_df.columns else None)

        # One fresh model per region and per fold: no weights are shared.
        model = CNN(in_channels=1)
        if args.mode != 'finetune':
            model.apply(initialize_weights)
        model = model.to(device)

        logger = get_logger(
            use_wandb=args.wandb, project=args.wandb_project,
            name=f'{args.region}_fold{cv_num}_{args.mode}',
            group=f'{args.region}_{args.mode}', job_type=f'fold{cv_num}', reinit=True,
        )

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = make_scheduler(optimizer, len(dataloader_train), args.lr_scheduler_choice, args.lr)

        trainer = CNN_Trainer(
            model=model,
            model_load_folder=args.model_load_folder,
            model_save_folder=str(model_dir),
            results_folder=str(results_dir),
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            dataloader_test=dataloader_test,
            epochs=args.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            cv_num=cv_num,
            region=args.region,
            valid_ids=valid_ids,
            device=device,
            patience=args.patience,
            logger=logger,
        )

        if args.mode == 'finetune':
            if not args.model_load_folder:
                raise ValueError('--model_load_folder is required for --mode finetune')
            load_epoch = args.load_epoch if args.load_epoch == 'best' else int(args.load_epoch)
            trainer.load(cv_num, load_epoch, gpu)
            # Reset optimizer (fresh state, no momentum carried over from pretraining)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            trainer.optimizer = optimizer
            trainer.scheduler = make_scheduler(
                optimizer, len(dataloader_train), args.lr_scheduler_choice, args.lr)

        start = time.time()
        trainer.train()
        print(f'Elapsed time for this fold: {(time.time() - start) / 60:.1f} minutes')

        # Promote the best epoch to cv-{fold}-best.pth.tar
        if trainer.valid_mae_list:
            best_epoch = int(np.argmin(trainer.valid_mae_list)) + 1
            best_src = model_dir / f'cv-{cv_num}-{best_epoch}.pth.tar'
            best_dst = model_dir / f'cv-{cv_num}-best.pth.tar'
            if best_src.exists():
                shutil.copy2(best_src, best_dst)
                print(f'[BEST] epoch {best_epoch} '
                      f'(val MAE={min(trainer.valid_mae_list):.4f}) -> {best_dst.name}')
            else:
                print(f'[WARN] best checkpoint not found at {best_src}')

        logger.finish()

    collect_oof_predictions(results_dir, args.region, args.n_splits,
                            results_dir / f'{args.region}_oof_predictions.csv')
    print('[DONE]')


if __name__ == '__main__':
    main()
