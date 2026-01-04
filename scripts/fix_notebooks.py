import json
import os

def update_notebook(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cell 1 is the import cell
    found_code = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            content = "".join(source)
            if "import sys" in content and "data.loader" in content:
                print(f"Found import cell in {path}")
                new_source = [
                    "import sys\n",
                    "import os\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import torch\n",
                    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
                    "\n",
                    "# Robust module path setup\n",
                    "def setup_sys_path():\n",
                    "    curr = os.getcwd()\n",
                    "    # Climb up until we find 'src'\n",
                    "    while curr != os.path.dirname(curr):\n",
                    "        if os.path.exists(os.path.join(curr, 'src')):\n",
                    "            if curr not in sys.path: sys.path.insert(0, curr)\n",
                    "            src = os.path.join(curr, 'src')\n",
                    "            if src not in sys.path: sys.path.insert(0, src)\n",
                    "            return curr\n",
                    "        curr = os.path.dirname(curr)\n",
                    "    return None\n",
                    "\n",
                    "project_root = setup_sys_path()\n",
                    "print(f\"Project root: {project_root}\")\n",
                    "\n",
                    "from data.loader import CWRUBearingDataLoader\n",
                    "from models.deep_learning import CNN1D\n",
                    "# Use project_root for data path\n",
                    "data_path = os.path.join(project_root, 'dataset1')\n",
                    "loader = CWRUBearingDataLoader(data_path)\n"
                ]
                
                if 'Hydraulique' in path:
                    new_source = [
                        "import sys\n",
                        "import os\n",
                        "from pathlib import Path\n",
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "\n",
                        "# Robust module path setup\n",
                        "def setup_sys_path():\n",
                        "    curr = os.getcwd()\n",
                        "    while curr != os.path.dirname(curr):\n",
                        "        if os.path.exists(os.path.join(curr, 'src')):\n",
                        "            if curr not in sys.path: sys.path.insert(0, curr)\n",
                        "            src = os.path.join(curr, 'src')\n",
                        "            if src not in sys.path: sys.path.insert(0, src)\n",
                        "            return curr\n",
                        "        curr = os.path.dirname(curr)\n",
                        "    return None\n",
                        "\n",
                        "project_root = setup_sys_path()\n",
                        "\n",
                        "from data.loader import HydraulicDataLoader\n",
                        "from data.features import FeatureExtractor\n",
                        "# Use project_root for data path\n",
                        "data_path = os.path.join(project_root, 'dataset0')\n",
                        "loader = HydraulicDataLoader(data_path)\n"
                    ]
                
                # Update the source but keep the rest of the cell content if it had more logic
                # For simplicity in this fix, we replace the first part.
                # But safer to just replace everything before the first import of our modules.
                cell['source'] = new_source + source[source.index(next(s for s in source if "loader =" in s)):] if any("loader =" in s for s in source) else new_source
                found_code = True
                break
    
    if found_code:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {path}")

# Paths to check
paths = [
    r'notebooks\01_Analyse_Exploratoire_Hydraulique.ipynb',
    r'notebooks\02_CWRU_Vibration_Deep_Learning.ipynb',
    r'notebooks\.ipynb_checkpoints\01_Analyse_Exploratoire_Hydraulique-checkpoint.ipynb',
    r'notebooks\.ipynb_checkpoints\02_CWRU_Vibration_Deep_Learning-checkpoint.ipynb'
]

for p in paths:
    update_notebook(p)

