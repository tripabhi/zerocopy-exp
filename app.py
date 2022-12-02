from flask import Flask, request
import ray
from modelhandler import ModelHandler

app = Flask(__name__)

handler = ModelHandler()


@app.route('/')
def welcome():
    return '<h1>Welcome to zerocopy experiments!</h2>'


@app.route('/serve', methods=["POST"])
def serve():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        return handler.handle(json)
    else:
        return 'Content-Type not supported!'


if __name__ == "__main__":
    app.run(debug=True)
