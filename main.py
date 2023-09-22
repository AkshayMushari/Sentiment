from unittest.mock import sentinel
from flask import Flask, json, jsonify, request
from Predict import predict
import requests
import ast

# dataset link -> https://mega.nz/file/Qahg1YiR#-svBlLcgRp5Jt01k9XMKQagftes33BhbOUBnuY29kQY

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World! this is sentimental analysis api</p>"


@app.route('/', methods=['GET', 'POST'])
def returnHappinessIndex():
    if request.method == 'POST':
        sentList = ast.literal_eval(request.args.get('sentList'))
        res = predict(sentList)
        result = {

            "total_score": res[0],
            "positive_score": res[1],
            "negative_score": res[2],
            "sentence_list": sentList,
        }
        return jsonify(result)
    else:
        return "<p>Please use proper API POST request call for happiness index</p>"


if __name__ == '__main__':
    app.run(host="localhost", port=5500, debug=True)
