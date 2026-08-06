import os
import pickle
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def save_checkpoint_patches(patches: List[Dict[str, Any]], filepath: str) -> bool:
    """Save raw Python NURBS patches to a pickle file for recovery."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(patches, f)
        logger.info(f"Saved {len(patches)} patches to checkpoint: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save patches checkpoint to {filepath}: {e}")
        return False

def load_checkpoint_patches(filepath: str) -> Optional[List[Dict[str, Any]]]:
    """Load raw Python NURBS patches from a pickle file to resume conversion."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            patches = pickle.load(f)
        logger.info(f"Loaded {len(patches)} patches from checkpoint: {filepath}")
        return patches
    except Exception as e:
        logger.error(f"Failed to load patches checkpoint from {filepath}: {e}")
        return None

def get_checkpoint_paths(base_input_path: str) -> dict:
    """Generate standardized checkpoint filenames based on the input file."""
    base_dir = os.path.dirname(base_input_path)
    base_name = os.path.splitext(os.path.basename(base_input_path))[0]
    
    return {
        'quads': os.path.join(base_dir, f"{base_name}_wip_quads.obj"),
        'patches': os.path.join(base_dir, f"{base_name}_wip_patches.pkl")
    }
