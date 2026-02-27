"""
GradTracer Auto-Compression Recipe Generator

The "Holy Grail" of model compression. Matches layer-specific training dynamics 
(gradient SNR, velocity, dead neurons) against available hardware compression techniques.
Produces a unified Mixed-Precision Quantization + Joint Pruning recipe.
"""
import json
from typing import Dict, Any

class RecipeGenerator:
    """
    Analyzes historical layer metrics from a tracking store and outputs an
    automated JSON compression recipe determining how aggressively each layer
    can be quantized and pruned without hurting predictive performance.
    """
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.store = tracker.store
        
    def generate(self, target_sparsity: float = 0.5) -> Dict[str, Any]:
        """
        Produce the joint compression recipe based on layer health and saliency.
        """
        from gradtracer.analyzers.health import layer_health_score, gradient_snr_per_layer
        
        health_scores = layer_health_score(self.store)
        snr_data = gradient_snr_per_layer(self.store)
        
        total_baseline_vram_mb = 0.0
        total_estimated_vram_mb = 0.0
        total_flops_baseline = 0.0
        total_flops_remaining = 0.0
        
        health_scores = layer_health_score(self.store)
        snr_data = gradient_snr_per_layer(self.store)
        
        recipe = {
            "metadata": {
                "target_sparsity": target_sparsity,
                "strategy": "Mixed-Precision Joint Pruning"
            },
            "layers": {}
        }
        
        for layer_name in self.store.layer_names:
            history = self.store.get_layer_history(layer_name)
            if not history:
                continue
                
            # Extract module metadata
            module_type = "Unknown"
            shape = []
            numel = 0
            if hasattr(self.tracker, "model"):
                # Clean name (e.g., 'layer1.0.conv1.weight' -> 'layer1.0.conv1')
                mod_name = layer_name.rsplit('.', 1)[0]
                try:
                    # In PyTorch, models can be accessed via get_submodule
                    mod = self.tracker.model.get_submodule(mod_name)
                    module_type = type(mod).__name__
                    param = getattr(mod, layer_name.split('.')[-1], None)
                    if param is not None:
                         shape = list(param.shape)
                         numel = param.numel()
                except Exception:
                    pass
            
            # Fallback if param extraction failed
            if numel == 0 and hasattr(history[-1], 'num_params'):
                numel = history[-1].num_params

            last_entry = history[-1]
            health = health_scores.get(layer_name, 100)
            snr = snr_data.get(layer_name, [0.0])[-1] if snr_data.get(layer_name) else 0.0
            
            # Rule Engine for Mixed-Precision & Pruning
            
            # 1. Critical Information Layers (High variance, active learning)
            if snr > 1.0 or health > 90:
                quant = "FP16"  # Preserve precision
                prune = 0.0     # Don't prune active parameters
                reason = "High gradient SNR; critical learning pathway."
                
            # 2. Dying or Dead Layers (Zero activations, stagnation)
            elif last_entry.dead_ratio > 0.5 or health < 30:
                quant = "INT4"  # Highest quantization
                prune = 0.8     # Aggressive structural pruning
                reason = "Severe stagnation or dead neurons detected."
                
            # 3. Dense but Low-Variance Layers (Feed-forward blocks, highly stable)
            else:
                quant = "INT8"  # Standard quantization
                prune = target_sparsity
                reason = "Stable, low-variance feed-forward representation."
                
            # Determine correct pruning strategy based on layer type
            prune_type = "unstructured_l1"
            if prune > 0:
                if "Conv" in module_type:
                    prune_type = "structured_channel"
                elif "Embedding" in module_type:
                    prune_type = "unstructured_l1"
                elif "Linear" in module_type:
                    if prune >= 0.5:
                        prune_type = "nvidia_2:4_structured"
                    else:
                        prune_type = "unstructured_l1"

            # Compute theoretical estimators
            baseline_bytes = numel * 4  # Assume starting at FP32
            total_baseline_vram_mb += baseline_bytes / (1024 * 1024)
            
            bytes_per_param = 2 if quant == "FP16" else (1 if quant == "INT8" else 0.5)
            retained_params = numel * (1.0 - prune)
            total_estimated_vram_mb += (retained_params * bytes_per_param) / (1024 * 1024)
            
            total_flops_baseline += numel
            total_flops_remaining += retained_params
                
            recipe["layers"][layer_name] = {
                "layer_type": module_type,
                "shape": shape,
                "quantization": quant,
                "prune_ratio": round(prune, 2),
                "prune_type": prune_type if prune > 0 else "none",
                "reason": reason,
                "health_score": round(health, 1),
                "dead_ratio": round(last_entry.dead_ratio, 2)
            }
            
        recipe["metadata"]["estimated_vram_saving_mb"] = round(total_baseline_vram_mb - total_estimated_vram_mb, 2)
        flops_reduction = 0.0
        if total_flops_baseline > 0:
            flops_reduction = 1.0 - (total_flops_remaining / total_flops_baseline)
        recipe["metadata"]["estimated_flops_reduction_ratio"] = round(flops_reduction, 3)
            
        return recipe
        
    def export_json(self, path: str = "gradtracer_recipe.json"):
        """Export the recipe to a JSON file for the VS Code Extension or deployment."""
        recipe = self.generate()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=4)
        return recipe
