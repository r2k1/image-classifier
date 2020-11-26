from imageai.Prediction import ImagePrediction
from urllib.request import urlopen
import os

threshold = 0
def image_prediction(image_file="2.jpg", model_file="resnet50_weights_tf_dim_ordering_tf_kernels.h5"):
  execution_path = os.getcwd()

  prediction = ImagePrediction()
  prediction.setModelTypeAsResNet()
  prediction.setModelPath(os.path.join(execution_path, model_file))
  prediction.loadModel()

  predictions, probabilities = prediction.predictImage(os.path.join(execution_path, image_file), result_count=3, input_type='file')
  result = []
  for eachPrediction, eachProbability in zip(predictions, probabilities):
    if eachProbability > threshold: 
      result.append({
        'key' : eachPrediction.replace('_', ' '),
        'probability' : eachProbability
      })
  return result


def image_prediction_by_url(image_url, model_file="resnet50_weights_tf_dim_ordering_tf_kernels.h5"):

  # download file to local file system
  f = urlopen(image_url)
  filename = os.path.basename(image_url)
  with open(filename, "wb") as local_file:
    local_file.write(f.read())

  #load prediction model
  execution_path = os.getcwd()
  prediction = ImagePrediction()
  prediction.setModelTypeAsResNet()
  prediction.setModelPath(os.path.join(execution_path, model_file))
  prediction.loadModel()

  predictions, probabilities = prediction.predictImage(os.path.join(execution_path, filename), result_count=5, input_type='file')
  result = []
  for eachPrediction, eachProbability in zip(predictions, probabilities):
    if eachProbability > threshold: 
      result.append({
        'key' : eachPrediction.replace('_', ' '),
        'probability' : eachProbability
      })
  os.remove(filename)
  return result

