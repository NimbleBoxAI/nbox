# /usr/bin/env python3
# -*- coding: utf-8 -*-

# start a simple webserver and serve json object at /

import nbox
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"message": "Hello, World!"}), 200


if __name__ == "__main__":
    app.run(debug=True)
