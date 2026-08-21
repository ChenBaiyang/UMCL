"""Unified runtime configuration system for UMCM.

This package intentionally keeps its top-level import surface light.
Import concrete helpers from submodules such as `config.resolver`
or `config.cli` to avoid pulling stage-3 runtime dependencies
during stage-1/stage-2 configuration bootstrap.
"""

__all__ = []
