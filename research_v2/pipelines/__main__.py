"""List available model implementations with: python3 -m research_v2.pipelines"""
import json
from . import MODEL_REGISTRY

if __name__ == "__main__":
    print(json.dumps({key: {
        "name": item.name,
        "implementation": item.implementation,
        "evidence": item.evidence,
        "generator_required": item.generator_required,
    } for key, item in MODEL_REGISTRY.items()}, indent=2))
