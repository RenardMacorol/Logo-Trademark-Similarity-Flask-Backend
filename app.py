import os
from flask import Flask, render_template
from flask_cors import CORS
import constant

# Modular Imports (Yung dati mo)
from core.model import WonksNetModel
from core.search_engine import LogoSearchEngine
from core.services import WonksNetService
from api.routes import api_bp
from web.routes import web_bp
from core.config import initialize_gpu

initialize_gpu()

app = Flask(__name__, static_folder='static')
CORS(app)  # Kahapon kailangan ito para sa connection

# Volumes check (Para sigurado lang tayo sa paths)
db_path = "/workspace/backend/static/database"
ds_path = "/workspace/backend/static/Dataset"
print(f"📁 Global: {'✅' if os.path.exists(db_path) else '❌'}")
print(f"📁 PH: {'✅' if os.path.exists(ds_path) else '❌'}")

WonksNetService.warm_up()

app.register_blueprint(web_bp)
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    # Siguraduhin na ang constant.HOST ay '0.0.0.0'
    # at ang port ay 5000
    app.run(host=constant.HOST, port=constant.PORT, debug=False)
