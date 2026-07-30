"""Region-wise brain age estimation (RegionBAE).

Public release of the pipeline used in the paper: atlas-defined regions are
masked on the fly, one independently parameterized CNN is trained per region
with k-fold cross-validation, and brain-age gaps are reported raw,
bias-corrected, and (optionally) inverse-normal transformed.
"""

from .regions import RegionConfig, is_global  # noqa: F401
from .dataset import Region_Dataset, build_region_mask  # noqa: F401
from .CNN import CNN, initialize_weights  # noqa: F401
from .CNN_Trainer import CNN_Trainer  # noqa: F401
from .utils import seed_everything, make_scheduler, get_logger  # noqa: F401
from .postprocess import (  # noqa: F401
    raw_bag,
    fit_bias_correction,
    apply_bias_correction,
    inverse_normal_transformation,
)

__version__ = '1.0.0'
