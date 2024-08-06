# -*- coding: utf-8 -*-
"""
Primary evaluator
--------------------

Resources package with primary evauator models.

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: MIT
"""
# Imports

# built-in
import importlib.resources

# local

# 3rd-party

# CSL


__package_path = importlib.resources.files(__name__)

BEST_FAITHFULNESS_CHECKPOINT = __package_path / 'best_faithfulness_checkpoint.pth'
LATEST_CHECKPOINT = __package_path / 'latest_checkpoint.pth'
RIDGE_FAITHFULNESS = __package_path / 'Ridge_PROB_CLS_MOTIONTEXT_Faithfulness.obj'
RIDGE_NATURALNESS = __package_path / 'Ridge_PROB_CLS_MOTIONTEXT_Naturalness.obj'
SVR_FAITHFULNESS = __package_path / 'SVR_PROB_CLS_MOTIONTEXT_Faithfulness.obj'
SVR_NATURALNESS = __package_path / 'SVR_PROB_CLS_MOTIONTEXT_Naturalness.obj'
TOKENIZER = __package_path / 'tokenizer.tk'
