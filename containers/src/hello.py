from flask import Flask
import numpy as np
import cv2
app = Flask(__name__)


@app.route('/')
def hello_world():
    return "<p>frats world</p>"
