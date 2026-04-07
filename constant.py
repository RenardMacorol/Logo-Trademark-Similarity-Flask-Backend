import os

# --- 1. BASE DIRECTORY ---
# This ensures paths work regardless of where the script is called from
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- 2. AI MODEL PATHS ---
# We use os.path.join for cross-platform compatibility (Arch vs Windows)
MODEL_FILE = os.path.join('test_models/initial_model.keras')

# --- 3. DATABASE PATHS ---
VECTORS_PATH = os.path.join(BASE_DIR, 'databases', 'registry_embeddings.npy')
METADATA_PATH = os.path.join(BASE_DIR, 'databases', 'registry_metadata.json')

# --- 4. HARDWARE SETTINGS ---
GPU_MEMORY_LIMIT = 3000  # Optimized for your GTX 1650
PORT = 5000
HOST = '0.0.0.0'
