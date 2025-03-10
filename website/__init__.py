from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'KEY'

    from .views import views
    app.register_blueprint(views, url_prefix='/')

    from .endpoint import endpoint_bp
    app.register_blueprint(endpoint_bp)

    return app
