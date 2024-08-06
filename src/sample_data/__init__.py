# -*- coding: utf-8 -*-
"""
sample_data.__init__
--------------------

Sample data for SPML model configuration

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: MIT
"""
# Imports

# built-in
import importlib.resources

__package_path = importlib.resources.files(__name__)

SAMPLE000000 = __package_path / '000000.npy'
MEAN = __package_path / 'Mean.npy'
STD = __package_path / 'Std.npy'