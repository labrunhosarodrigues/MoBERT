# -*- coding: utf-8 -*-
"""
configs
----------------

Sub-package with configuration utils.

:copyright: (c) 2024 by Cognitive Systems Lab.
:license: MIT
"""
# Imports

# built-in
import importlib.resources

# local

# 3rd-party


BASE_CONFIG = importlib.resources.files(__package__) / 'base_config.yml'

