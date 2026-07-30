"""Region configuration: atlas file + region name -> atlas label(s).

This is the only part of the pipeline that is *not* a copy of the original
research code: the original hard-coded one atlas path and a fixed
``{'caudate': 1, ..., 'thalamus': 9}`` mapping. The behaviour is unchanged; the
mapping simply moved into a JSON file so a different atlas / label combination
can be used without editing code.

JSON format (see ``configs/regions_mni_structural.json``)::

    {
      "atlas": "/path/to/atlas.nii.gz",
      "dilate_radius": 4,
      "regions": {
        "global": null,
        "caudate": 1,
        "frontal_lobe": [3, 12]
      }
    }

``null`` means whole brain (no masking); an integer selects one atlas label; a
list selects the union of several labels. Only the regions listed in the config
exist — nothing is added implicitly, so a config with a single whole-brain entry
trains exactly one model. ``atlas`` may be omitted and supplied on the command
line instead.
"""

import json
import os


#: Region names treated as "whole brain, no mask" when they carry no labels.
#: ``imgs`` is the name used by the original research code and is kept so
#: existing cohort CSVs / configs keep working.
GLOBAL_ALIASES = ('global', 'imgs', 'whole_brain', 'wholebrain')

DEFAULT_DILATE_RADIUS = 4


def is_global(region):
    """True when ``region`` denotes the unmasked whole brain."""
    return region is None or str(region).lower() in GLOBAL_ALIASES


class RegionConfig:
    """Atlas path + ``{region_name: [atlas labels] or None}`` + dilation radius."""

    def __init__(self, atlas_path=None, regions=None, dilate_radius=DEFAULT_DILATE_RADIUS):
        self.atlas_path = atlas_path
        self.dilate_radius = int(dilate_radius)
        self.regions = {}

        for name, labels in (regions or {}).items():
            if labels is None:
                self.regions[name] = None
            elif isinstance(labels, (list, tuple)):
                self.regions[name] = [int(v) for v in labels]
            else:
                self.regions[name] = [int(labels)]

        if any(v is not None for v in self.regions.values()):
            if not self.atlas_path:
                raise ValueError('An atlas path is required for non-global regions')
            if not os.path.exists(self.atlas_path):
                raise FileNotFoundError(f'Atlas not found: {self.atlas_path}')

    @classmethod
    def from_json(cls, path, atlas_path=None, dilate_radius=None):
        with open(path) as f:
            cfg = json.load(f)
        if 'regions' not in cfg:
            raise KeyError(f'{path}: config must contain a "regions" object')
        return cls(
            atlas_path=atlas_path or cfg.get('atlas'),
            regions=cfg['regions'],
            dilate_radius=(dilate_radius if dilate_radius is not None
                           else cfg.get('dilate_radius', DEFAULT_DILATE_RADIUS)),
        )

    @property
    def names(self):
        """Region names exactly as defined in the config file, in file order."""
        return list(self.regions)

    @property
    def masked_names(self):
        """Region names that actually get an atlas mask (excludes whole brain)."""
        return [n for n, v in self.regions.items() if v is not None]

    def labels_for(self, region):
        """Atlas labels for ``region``; ``None`` means whole brain (no mask)."""
        if region in self.regions:
            return self.regions[region]
        if is_global(region):
            return None
        raise KeyError(f"Unknown region '{region}'. Config defines: {self.names}")

    def __repr__(self):
        return (f'RegionConfig(atlas={self.atlas_path!r}, '
                f'dilate_radius={self.dilate_radius}, regions={self.names})')
