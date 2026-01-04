import sys
import os
import numpy as np

# Simulate the notebook's environment
# The notebook is in notebooks/.ipynb_checkpoints/
# So CWD should be there for a fair test.
target_dir = r'c:\Users\MSI\Desktop\VibroFlow AI\notebooks\.ipynb_checkpoints'
os.chdir(target_dir)
print(f"CWD set to: {os.getcwd()}")

# The new robust path setup from fix_notebooks.py
def setup_sys_path():
    curr = os.getcwd()
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, 'src')):
            if curr not in sys.path: sys.path.insert(0, curr)
            src = os.path.join(curr, 'src')
            if src not in sys.path: sys.path.insert(0, src)
            return curr
        curr = os.path.dirname(curr)
    return None

project_root = setup_sys_path()
print(f"Project root found: {project_root}")
print(f"sys.path[0]: {sys.path[0]}")
print(f"sys.path[1]: {sys.path[1]}")

try:
    from data.loader import CWRUBearingDataLoader
    from models.deep_learning import CNN1D
    print("Success: Imports worked!")
    
    # Test data loading
    loader = CWRUBearingDataLoader('../../dataset1')
    X, y = loader.load_preprocessed_npz()
    print(f"Success: Data loaded! Shape: {X.shape}")
except ImportError as e:
    print(f"Failure (Import): {e}")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Failure (File): {e}")
    sys.exit(1)
except Exception as e:
    print(f"Failure (General): {e}")
    sys.exit(1)
