# -*- coding: utf-8 -*-
"""
Primary evaluator
--------------------

Resources package with primary evauator models.

Resources should be loadded using `get_file_data(RESOURCE)`, which
loads the file contents as a BytesIO buffer, which can then be
passed to the appropriate file parsing function to load the resource object.

Resources provided in this module are:
 - BEST_FAITHFULNESS_CHECKPOINT: pytorch weights for best faithfulness checkpoint.
 - LATEST_CHECKPOINT: pytorch weights for the lastest checkpoint.
 - RIDGE_FAITHFULNESS: pickled RIDGE regressor for faithfulness.
 - RIDGE_NATURALNESS: pickled RIDGE regressor for naturalness.
 - SVR_FAITHFULNESS: pickled SVR regressor for faithfulness.
 - SVR_NATURALNESS: pickled SVR regressor for naturalness.
 - TOKENIZER: Serialized tokenizers.Tokenizer object.

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: MIT
"""
# Imports

# built-in
import io
import importlib.resources
from pathlib import Path
import os

# local

# 3rd-party
from zipfile import ZipFile
# CSL


__package_path = importlib.resources.files(__name__)

ARCHIVE = __package_path / 'archive.zip'
BASE_ARCHIVE = Path("primary_evaluator/std_bpe2000")
BEST_FAITHFULNESS_CHECKPOINT = BASE_ARCHIVE / 'best_faithfulness_checkpoint.pth'
LATEST_CHECKPOINT = BASE_ARCHIVE / 'latest_checkpoint.pth'
RIDGE_FAITHFULNESS = BASE_ARCHIVE / 'Ridge_PROB_CLS_MOTIONTEXT_Faithfulness.obj'
RIDGE_NATURALNESS = BASE_ARCHIVE / 'Ridge_PROB_CLS_MOTIONTEXT_Naturalness.obj'
SVR_FAITHFULNESS = BASE_ARCHIVE / 'SVR_PROB_CLS_MOTIONTEXT_Faithfulness.obj'
SVR_NATURALNESS = BASE_ARCHIVE / 'SVR_PROB_CLS_MOTIONTEXT_Naturalness.obj'
TOKENIZER = BASE_ARCHIVE / 'tokenizer.tk'


def get_file_data(resource):
    """
    Load ByteIO buffer with files in archive data.
    """
    if not os.path.exists(ARCHIVE):
        raise FileNotFoundError(
            """No archive found at expected location.
            Please download it from:
            https://drive.usercontent.google.com/download?id=1gmljNRJKf_IujUIlcmCl9Q6mZI_Qceiv&export=download&authuser=0

            And run the function `save_primary_evaluator_archive(path_to_downloaded_file)`
            to properly install the archive."""
        )
    with ZipFile(ARCHIVE) as zip:
        with zip.open(resource) as fid:
            data = io.BytesIO(fid.read())
    
    return data


def save_primary_evaluator_archive(zip_path):
    """
    Properly save ZIP archive as a resource in this package.
    """

    os.rename(zip_path, ARCHIVE)
