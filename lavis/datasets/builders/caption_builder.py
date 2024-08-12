"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry

from lavis.datasets.datasets.tfr_image_text_dialog_datasets import (
    ImageTextDialogDataset,
)

@registry.register_builder("msvd")
class MSVDBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextDialogDataset
    eval_dataset_cls = None #! TODO

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/caption/msvd.yaml",
    }

