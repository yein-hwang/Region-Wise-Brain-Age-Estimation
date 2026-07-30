#!/usr/bin/env python
"""predict.py - run one region's fold models over a cohort and write raw BAG.

Generalized from the research code (``scripts/10_inference_brain_age.py``). All
fold models are held in memory and every batch is passed through each of them,
so the per-fold predictions are always available for provenance.

The final prediction per row follows the ``prediction_mode`` column when the
cohort table has one:

  ``oof_fold_K``  use fold K's model only — the honest out-of-fold prediction
                  for a subject that was in fold K's validation set
  ``ensemble``    mean over all fold models — for rows no fold model saw

Without that column every row is scored with ``--default_mode`` (ensemble).

Output columns: subjectID, chronological_age, predicted_age, raw_bag,
prediction_mode, fold_used, region, pred_fold_0..N, plus any columns listed in
``--keep_cols``.
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regionbae import CNN, RegionConfig, Region_Dataset, seed_everything  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort_csv', required=True)
    p.add_argument('--region', required=True)
    p.add_argument('--region_config', required=True)
    p.add_argument('--atlas', default=None)
    p.add_argument('--dilate_radius', type=int, default=None)
    p.add_argument('--model_dir', required=True,
                   help='directory with cv-{fold}-best.pth.tar for every fold')
    p.add_argument('--checkpoint_pattern', default='cv-{fold}-best.pth.tar')
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--output_csv', required=True)
    p.add_argument('--default_mode', default='ensemble',
                   help="mode for rows without a prediction_mode value ('ensemble' or 'oof_fold_K')")
    p.add_argument('--keep_cols', nargs='*', default=[],
                   help='extra cohort columns to carry into the output')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--image_col', default='imgs')
    p.add_argument('--age_col', default='age')
    p.add_argument('--id_col', default='subjectID')
    return p.parse_args()


def load_fold_model(path, device):
    """Build CNN, load weights, strip a 'module.' prefix if present."""
    model = CNN(in_channels=1).to(device)
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_state[k.replace('module.', '') if k.startswith('module.') else k] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    gpu = bool(args.gpu) and torch.cuda.is_available()
    device = torch.device('cuda' if gpu else 'cpu')
    seed_everything(args.seed)

    region_config = RegionConfig.from_json(
        args.region_config, atlas_path=args.atlas, dilate_radius=args.dilate_radius)
    df = pd.read_csv(args.cohort_csv)

    print('=' * 60)
    print(f'[predict] region={args.region} labels={region_config.labels_for(args.region)}')
    print(f'  cohort     : {args.cohort_csv} (n={len(df)})')
    print(f'  model_dir  : {args.model_dir} ({args.n_folds} folds)')
    print(f'  device     : {device}')
    print('=' * 60)

    dataset = Region_Dataset(
        df, None, args.region, region_config=region_config,
        image_col=args.image_col, age_col=args.age_col, id_col=args.id_col)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset),
                        collate_fn=dataset.collate_fn, pin_memory=gpu, num_workers=args.n_workers)

    model_dir = Path(args.model_dir)
    models = {}
    for fold in range(args.n_folds):
        path = model_dir / args.checkpoint_pattern.format(fold=fold)
        if not path.exists():
            raise FileNotFoundError(f'Missing fold {fold} checkpoint: {path}')
        models[fold] = load_fold_model(str(path), device)
        print(f'  loaded fold {fold}: {path.name}')

    preds = {fold: [] for fold in models}
    seen, total = 0, len(dataset)
    report_every = max(1, total // 20)
    with torch.no_grad():
        for batch_imgs, _ in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            for fold, model in models.items():
                out = model(batch_imgs)
                preds[fold].extend(out.detach().cpu().numpy().flatten().tolist())
            prev, seen = seen, seen + batch_imgs.size(0)
            if seen // report_every > prev // report_every or seen == total:
                print(f'  predicted {seen}/{total}', flush=True)

    pred_cols = {f'pred_fold_{f}': np.asarray(preds[f], dtype=np.float64) for f in models}
    for name, arr in pred_cols.items():
        if len(arr) != len(df):
            raise ValueError(f'{name} length mismatch: {len(arr)} vs {len(df)}')

    if 'prediction_mode' in df.columns:
        pmode = df['prediction_mode'].fillna(args.default_mode).astype(str).values
    else:
        pmode = np.array([args.default_mode] * len(df))

    final_pred = np.zeros(len(df), dtype=np.float64)
    fold_used = np.empty(len(df), dtype=object)
    for i, m in enumerate(pmode):
        if m.startswith('oof_fold_'):
            fi = int(m.split('_')[-1])
            if fi not in models:
                raise ValueError(f'Row {i}: prediction_mode {m!r} refers to an unavailable fold')
            final_pred[i] = pred_cols[f'pred_fold_{fi}'][i]
            fold_used[i] = fi
        elif m == 'ensemble':
            final_pred[i] = np.mean([pred_cols[f'pred_fold_{f}'][i] for f in models])
            fold_used[i] = 'ensemble'
        else:
            raise ValueError(f'Unknown prediction_mode at row {i}: {m!r}')

    age = df[args.age_col].astype(float).values
    out = pd.DataFrame({
        'subjectID': df[args.id_col] if args.id_col in df.columns else np.arange(len(df)),
        'region': args.region,
        'chronological_age': age,
        'predicted_age': final_pred,
        'raw_bag': final_pred - age,
        'prediction_mode': pmode,
        'fold_used': fold_used,
    })
    for col in args.keep_cols:
        if col in df.columns:
            out[col] = df[col].values
        else:
            print(f'[WARN] --keep_cols column not in cohort: {col}')
    for fold in models:
        out[f'pred_fold_{fold}'] = pred_cols[f'pred_fold_{fold}']

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f'\nWritten: {out_path} ({len(out)} rows)')
    print(f'raw BAG: mean={out["raw_bag"].mean():.3f} sd={out["raw_bag"].std():.3f} '
          f'MAE={out["raw_bag"].abs().mean():.3f}')
    print('[DONE]')


if __name__ == '__main__':
    main()
