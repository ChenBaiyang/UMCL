"""
Hierarchical Learning Rate Optimizer setup for UMCL Five-View Model
"""
from typing import Any

import torch
import torch.nn as nn


def build_hierarchical_optimizer(
    model: nn.Module,
    *,
    backbone_lr: float,
    backbone_weight_decay: float,
    top_layer_lr: float,
    top_layer_weight_decay: float,
    head_lr: float,
    head_weight_decay: float,
    optimizer_class: type = torch.optim.AdamW,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Build a hierarchical optimizer with different learning rates for different model components.
    
    Args:
        model: The UMCL Five-View model
        backbone_lr: Learning rate for Transformer backbones
        backbone_weight_decay: Weight decay for Transformer backbones
        top_layer_lr: Learning rate for encoder top layers
        top_layer_weight_decay: Weight decay for encoder top layers
        head_lr: Learning rate for classification/reconstruction heads
        head_weight_decay: Weight decay for classification/reconstruction heads
        optimizer_class: Optimizer class to use (default: AdamW)
        **optimizer_kwargs: Additional kwargs for optimizer (like momentum for SGD)
    
    Returns:
        Optimizer with hierarchical parameter groups
    """
    
    backbone_params = []
    top_layer_params = []
    head_params = []
    
    # Categorize parameters based on module type
    # Identify head/posterior modules that should use high learning rate
    head_module_keywords = {"posterior", "head", "classifier", "scorer"}
    
    # First pass: categorize modules
    processed_module_params = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Module) or list(module.parameters()) == []:
            continue
        
        # Get module parameters (direct children only to avoid duplication)
        if len(list(module.children())) == 0:  # Leaf module
            module_params = list(module.parameters())
            for p in module_params:
                processed_module_params.add(id(p))
            
            # Classify based on module name and hierarchy
            name_lower = name.lower()
            
            # HEAD MODULES (posterior_heads, reconstruction_heads, task_heads, etc.)
            if any(keyword in name_lower for keyword in head_module_keywords):
                head_params.extend(module_params)
            # BACKBONE MODULES (transformer models, backbone, shared_backbone)
            elif any(keyword in name_lower for keyword in ["backbone", "transformer"]):
                # Exclude if it's clearly a top-level aggregator or encoder wrapper
                if "encoder" not in name_lower or "backbone" in name_lower:
                    backbone_params.extend(module_params)
                else:
                    top_layer_params.extend(module_params)
            # TOP LAYER MODULES (projections, MLPs, embeddings, aggregators)
            elif any(keyword in name_lower for keyword in [
                "projection", "mlp", "linear", "embedding", "norm", "fusion",
                "aggregator", "context", "encoder", "pooling"
            ]):
                top_layer_params.extend(module_params)
            else:
                # Default to top_layer for unknown modules
                top_layer_params.extend(module_params)
    
    # Second pass: process remaining parameters (like nn.Parameter)
    for name, param in model.named_parameters():
        if id(param) in processed_module_params:
            continue
        
        # Classify based on parameter name
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in head_module_keywords):
            head_params.append(param)
        elif any(keyword in name_lower for keyword in ["backbone", "transformer"]):
            backbone_params.append(param)
        else:
            # Default to head for unclassified remaining parameters
            head_params.append(param)
    
    # Remove duplicates while preserving order
    def deduplicate_params(param_list):
        seen = set()
        deduped = []
        for p in param_list:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                deduped.append(p)
        return deduped
    
    backbone_params = deduplicate_params(backbone_params)
    top_layer_params = deduplicate_params(top_layer_params)
    head_params = deduplicate_params(head_params)
    
    # Remove duplicates across groups (prioritize: backbone > head > top_layer)
    backbone_ids = set(id(p) for p in backbone_params)
    head_ids = set(id(p) for p in head_params)
    
    # Remove backbone params from head
    head_params = [p for p in head_params if id(p) not in backbone_ids]
    head_ids = set(id(p) for p in head_params)
    
    # Remove backbone and head params from top_layer
    top_layer_params = [p for p in top_layer_params if id(p) not in backbone_ids and id(p) not in head_ids]
    top_layer_ids = set(id(p) for p in top_layer_params)
    
    # Check for any parameters not yet in a group
    all_grouped_ids = backbone_ids | head_ids | top_layer_ids
    remaining_params = []
    remaining_param_info = []
    for name, param in model.named_parameters():
        if id(param) not in all_grouped_ids and param.requires_grad:
            remaining_params.append(param)
            remaining_param_info.append((name, param.shape))
    
    # Build parameter groups
    param_groups = []
    
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": backbone_weight_decay,
            "name": "backbone",
        })
    
    if top_layer_params:
        param_groups.append({
            "params": top_layer_params,
            "lr": top_layer_lr,
            "weight_decay": top_layer_weight_decay,
            "name": "top_layer",
        })
    
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": head_weight_decay,
            "name": "head",
        })
    
    # Add any remaining parameters
    if remaining_params:
        param_groups.append({
            "params": remaining_params,
            "lr": head_lr,
            "weight_decay": head_weight_decay,
            "name": "remaining",
        })
    
    print(f"\n{'='*80}")
    print(f"Hierarchical Optimizer Parameter Groups:")
    print(f"{'='*80}")
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        print(f"  {group['name']:25s}: {len(group['params']):5d} params | "
              f"lr={group['lr']:.2e} | wd={group['weight_decay']:.2e}")
    
    # Show remaining parameters info
    if remaining_param_info:
        print(f"\n{'='*80}")
        print(f"Remaining Parameters (not yet classified):")
        print(f"{'='*80}")
        for param_name, param_shape in remaining_param_info:
            print(f"  {param_name:50s} | shape={param_shape}")
    
    print(f"{'='*80}\n")
    
    # Create optimizer
    optimizer = optimizer_class(param_groups, **optimizer_kwargs)
    
    return optimizer
