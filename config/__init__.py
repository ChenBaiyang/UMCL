"""Unified runtime configuration system for UMCL.

This package intentionally keeps its top-level import surface light.
Import concrete helpers from submodules such as `config.resolver` to avoid
pulling stage-3 runtime dependencies during stage-1/stage-2 bootstrap.
"""

__all__ = []
