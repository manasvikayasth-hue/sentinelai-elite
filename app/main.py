from flask import Flask
from app.routes.detect_route import detect_bp


def create_app():
    app = Flask(__name__)

    # ✅ define root route INSIDE function
    @app.route("/")
    def home():
        return {
            "status": "running",
            "message": "Border Surveillance AI is live 🚀"
        }

    # Register routes
    app.register_blueprint(detect_bp, url_prefix="/api")

    return app