# -*- coding: utf-8 -*-
"""Per-survey constants and conventions.

Everything in `ggt/` that needs to know a plate scale, a zeropoint, a band
name or a redshift binning imports it from a module in here, and from
nowhere else. Adding a survey means adding a module beside `euclid.py`; it
should not mean editing the data, model or training code.

Modules are imported by full path (`from ggt.surveys import euclid`) rather
than re-exported, so that adding a survey cannot break an existing one.
"""
