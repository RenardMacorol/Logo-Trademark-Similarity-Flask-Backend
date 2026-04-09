import os

# --- 1. BASE DIRECTORY ---
# This ensures paths work regardless of where the script is called from
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- 2. AI MODEL PATHS ---
# We use os.path.join for cross-platform compatibility (Arch vs Windows)
# MODEL_FILE = os.path.join('test_models/initial_model.keras')
# MODEL_FILE = os.path.join('test_models/Ready2DeployModel.keras')
MODEL_FILE = os.path.join('test_models/Ready2DeployModelV2.keras')

# --- 3. DATABASE PATHS ---
# VECTORS_PATH = os.path.join(BASE_DIR, 'databases',
#                            'version1/registry_embeddings.npy')
# METADATA_PATH = os.path.join(
#    BASE_DIR, 'databases', 'version1/registry_metadata.json')

# VECTORS_PATH = os.path.join(BASE_DIR, 'databases',
#                            'version2/registry_embeddings.npy')
# METADATA_PATH = os.path.join(
#    BASE_DIR, 'databases', 'version2/registry_metadata.json')

# DB PATHS SCOPE
DB_BOTH_SCOPE = 'BOTH'
DB_PH_SCOPE = 'PH'
DB_GLOBAL_SCOPE = 'GLOBAL'

# DB Directories

PH_VECTORS_PATH = os.path.join(BASE_DIR, 'databases',
                               'version3/PhDataset/registry_embeddings.npy')
PH_METADATA_PATH = os.path.join(
    BASE_DIR, 'databases', 'version3/PhDataset/registry_metadata.json')

GLOBAL_VECTORS_PATH = os.path.join(BASE_DIR, 'databases',
                                   'version3/GlobalDataset/logo2k_embeddings.npy')
GLOBAL_METADATA_PATH = os.path.join(
    BASE_DIR, 'databases', 'version3/GlobalDataset/logo2k_metadata.json')

# --- 4. HARDWARE SETTINGS ---
GPU_MEMORY_LIMIT = 3000  # Optimized for your GTX 1650
PORT = 5000
HOST = '0.0.0.0'
