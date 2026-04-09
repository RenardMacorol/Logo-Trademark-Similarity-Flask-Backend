# Global Imports
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import constant

# Modular Imports
from core.model import WonksNetModel
from core.search_engine import LogoSearchEngine
from core.services import WonksNetService
from processor.image_utils import auto_crop_logo, encode_to_base64
from api.routes import api_bp
from web.routes import web_bp
from core.config import initialize_gpu

# --- 1. GPU CONFIG (GTX 1650 Optimized) ---
initialize_gpu()

app = Flask(__name__)

# --- 2. INITIALIZE SERVICES (Eager Loading) ---
print("🚀 [SERVER] Initiating WonksNet Engine...")

# CHANGE: Instead of get_engine(), use warm_up()
# This ensures both PH and GLOBAL are in RAM before any user connects
WonksNetService.warm_up()

# --- 3. REGISTER BLUEPRINTS ---
# Web UI at root (/)
app.register_blueprint(web_bp)

# Mobile API at (/api)
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    # host 0.0.0.0 is necessary for Docker/Dashboard-1 access
    app.run(host=constant.HOST, port=constant.PORT, debug=False)
