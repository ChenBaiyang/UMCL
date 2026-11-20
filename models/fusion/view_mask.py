"""
View masking utilities for selective testing of individual modalities.
"""
from __future__ import annotations

from typing import Set, Dict, Any


VIEW_ORDER = ["content", "stance", "propagation", "evidence", "source"]
VIEW_SET = set(VIEW_ORDER)


class ViewMask:
    """Controls which views participate in forward pass, gradients, and loss computation"""
    
    def __init__(self, active_views: Set[str] | None = None, all_views: Set[str] | None = None):
        """
        Args:
            active_views: Set of view names to activate (e.g., {"content", "stance"})
                         If None, all views are active by default
            all_views: Set of all available view names (default: {"content", "stance", "propagation", "evidence", "source"})
        """
        self.all_views = all_views or VIEW_SET
        self.active_views = active_views or self.all_views
        
        # Validate
        invalid = self.active_views - self.all_views
        if invalid:
            raise ValueError(f"Invalid view names: {invalid}. Valid views: {self.all_views}")
    
    def is_active(self, view_name: str) -> bool:
        """Check if a view is active"""
        return view_name in self.active_views
    
    def get_active_views(self) -> list[str]:
        """Get list of active views in consistent order"""
        return [v for v in VIEW_ORDER if v in self.active_views]
    
    def get_inactive_views(self) -> list[str]:
        """Get list of inactive views"""
        return [v for v in VIEW_ORDER if v not in self.active_views]
    
    def filter_dict_by_views(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Filter dictionary to only contain active views"""
        return {k: v for k, v in d.items() if k in self.active_views}
    
    def __repr__(self) -> str:
        active_str = ", ".join(self.get_active_views())
        inactive_str = ", ".join(self.get_inactive_views())
        pct = 100 * len(self.active_views) / len(self.all_views)
        return f"ViewMask(active=[{active_str}], inactive=[{inactive_str}], {pct:.0f}% coverage)"
    
    def __bool__(self) -> bool:
        """True if all views are active"""
        return len(self.active_views) == len(self.all_views)


def parse_views_string(views_str: str | None) -> Set[str] | None:
    """
    Parse view specification string
    
    Examples:
        "content,stance" -> {"content", "stance"}
        "all" -> None (means all views)
        "content;stance;propagation" -> {"content", "stance", "propagation"}
    """
    if not views_str or views_str.lower() == "all":
        return None
    
    # Try both comma and semicolon separators
    views_str = views_str.replace(";", ",")
    views = {v.strip() for v in views_str.split(",") if v.strip()}
    return views if views else None


def create_view_mask_from_config(view_config: str | None) -> ViewMask:
    """Create ViewMask from configuration string
    
    Args:
        view_config: Configuration string like "content,stance" or None for all views
    
    Returns:
        ViewMask instance
    """
    if view_config:
        preset = PRESET_MASKS.get(view_config.strip().lower())
        if preset is not None:
            return ViewMask(active_views=set(preset.active_views))
    active_views = parse_views_string(view_config)
    return ViewMask(active_views=active_views)


# Preset masks for common testing scenarios
PRESET_MASKS = {
    "all": ViewMask(),
    "content_only": ViewMask(active_views={"content"}),
    "stance_only": ViewMask(active_views={"stance"}),
    "propagation_only": ViewMask(active_views={"propagation"}),
    "evidence_only": ViewMask(active_views={"evidence"}),
    "source_only": ViewMask(active_views={"source"}),
    "content_stance": ViewMask(active_views={"content", "stance"}),
    "content_propagation": ViewMask(active_views={"content", "propagation"}),
    "no_stance": ViewMask(active_views={"content", "propagation", "evidence", "source"}),
    "no_propagation": ViewMask(active_views={"content", "stance", "evidence", "source"}),
}


def get_mask_name(mask: ViewMask) -> str:
    """Get a human-readable name for a mask"""
    for name, preset in PRESET_MASKS.items():
        if mask.active_views == preset.active_views:
            return name
    return f"custom({','.join(mask.get_active_views())})"
