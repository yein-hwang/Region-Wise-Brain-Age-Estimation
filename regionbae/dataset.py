"""Whole-brain MRI loader with on-the-fly region masking via an atlas.

Copied from the research code (``src_v2/dataset.py``) with one generalization:
the region -> atlas-label mapping and the atlas path come from a
:class:`~regionbae.regions.RegionConfig` instead of a module-level constant.
Mask construction itself (label selection -> binary dilation on the atlas grid
-> nearest-neighbour resample to the image grid -> ``image[~mask] = 0``) is
unchanged.

No intensity normalization, cropping or resizing happens here: the loader feeds
raw float voxels to the network, exactly as in the original code. Images are
expected to be preprocessed beforehand (see README).
"""

import threading

import ants
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .regions import RegionConfig

# Process-level cache: dilated bool masks are deterministic in
# (atlas, labels, radius, reference grid), so we compute each one at most once
# even when multiple Dataset instances (train + valid + test) request the same
# region.
_MASK_CACHE = {}
_MASK_CACHE_LOCK = threading.Lock()


def build_region_mask(atlas_path, labels, dilate_radius, reference_image_path=None):
    """Return a contiguous bool ndarray for the given atlas labels, cached by key.

    Dilation is performed in the atlas's native grid (e.g. 1 mm MNI), then — if
    ``reference_image_path`` is given — the result is nearest-neighbor resampled
    to that image's grid. This lets datasets whose images live on a downsampled
    MNI grid (e.g. 128^3) reuse the same 1 mm atlas.
    """
    labels = tuple(int(v) for v in labels)
    key = (str(atlas_path), labels, int(dilate_radius), reference_image_path)
    with _MASK_CACHE_LOCK:
        if key in _MASK_CACHE:
            return _MASK_CACHE[key]

    atlas_ants = ants.image_read(str(atlas_path))
    if len(labels) == 1:
        selected = (atlas_ants == labels[0]).numpy()
    else:
        selected = np.isin(atlas_ants.numpy(), labels)
    if not selected.any():
        raise ValueError(f'Labels {list(labels)} are absent from atlas {atlas_path}')

    # ANTs binary morphology silently no-ops on unsigned-char input; cast to float.
    region_mask_ants = atlas_ants.new_image_like(selected.astype('float32'))
    dilated_ants = ants.morphology(
        region_mask_ants, radius=int(dilate_radius), operation='dilate', mtype='binary'
    )

    if reference_image_path is not None:
        reference_ants = ants.image_read(str(reference_image_path))
        dilated_ants = ants.resample_image_to_target(
            dilated_ants, reference_ants, interp_type='nearestNeighbor'
        )

    mask_bool = np.ascontiguousarray(dilated_ants.numpy().astype(bool))

    with _MASK_CACHE_LOCK:
        _MASK_CACHE.setdefault(key, mask_bool)
        return _MASK_CACHE[key]


class Region_Dataset(Dataset):
    """Whole-brain MRI loader with on-the-fly region masking via an atlas.

    Always reads the whole-brain T1w volume from ``image_col``. When the region
    has atlas labels attached, the label mask is dilated and each sample is
    masked in-place (``image[~mask] = 0``). The dilated mask is cached at module
    level so train/valid/test datasets sharing the same (atlas, labels, radius)
    reuse a single mask.
    """

    def __init__(
        self,
        mri_csv,
        indices=None,
        roi=None,
        region_config=None,
        image_col='imgs',
        age_col='age',
        id_col='subjectID',
    ):
        super().__init__()
        self.data_csv = mri_csv.reset_index(drop=True)
        self.roi = roi
        self.image_col = image_col
        self.age_col = age_col
        self.id_col = id_col
        self.indices = list(indices) if indices is not None else list(range(len(self.data_csv)))

        for col in (image_col, age_col):
            if col not in self.data_csv.columns:
                raise KeyError(f'Cohort CSV is missing required column "{col}"')

        self.image_paths = [self.data_csv.loc[i, image_col] for i in self.indices]
        self.ages = [self.data_csv.loc[i, age_col] for i in self.indices]
        self.subject_ids = (
            [self.data_csv.loc[i, id_col] for i in self.indices]
            if id_col in self.data_csv.columns else None
        )

        labels = None
        if region_config is not None:
            if not isinstance(region_config, RegionConfig):
                raise TypeError('region_config must be a RegionConfig instance')
            labels = region_config.labels_for(roi)

        if labels is None:
            self.mask_bool = None
        else:
            # Subjects share the same template grid, so the first image suffices
            # as a resample reference (lets the 1 mm atlas mask match downsampled
            # data grids like 128^3).
            reference_image_path = self.image_paths[0] if self.image_paths else None
            self.mask_bool = build_region_mask(
                region_config.atlas_path, labels, region_config.dilate_radius,
                reference_image_path,
            )

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0).contiguous()
        ages = torch.stack(labels, dim=0).float()
        return images, ages

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        age = float(self.ages[index])

        image = nib.load(image_path).get_fdata(dtype='float32')

        if self.mask_bool is not None:
            # Image shape must match the atlas grid (subjects are registered).
            if image.shape != self.mask_bool.shape:
                raise ValueError(
                    f'Image shape {image.shape} does not match atlas mask shape '
                    f'{self.mask_bool.shape} for {image_path}'
                )
            # In-place zero-out outside the dilated region (no multiplication, no extra array).
            image[~self.mask_bool] = 0

        image = torch.from_numpy(image)
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        image = image.unsqueeze(0)
        age = torch.tensor(age, dtype=torch.float32)
        return image, age
