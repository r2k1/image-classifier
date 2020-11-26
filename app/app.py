import json, base64
from flask import Flask, redirect, url_for, request, jsonify
from prediction import image_prediction, image_prediction_by_url


app = Flask(__name__)

@app.route('/image-classifier', methods = ['GET', 'POST'])
def classifier():
  if request.method == 'GET':
    ret = image_prediction()
  else:
    data = json.loads(request.data)
    image_url = data["image_url"]
    ret = image_prediction_by_url(image_url)
  
  return jsonify(ret)

if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0')