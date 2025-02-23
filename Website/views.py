from flask import Flask, Blueprint, render_template

app = Flask(__name__)

views = Blueprint("views", __name__)


@views.route("/")
def home():
    return render_template("index.html")


@views.route("/about")
def about():
    return render_template("about.html")
