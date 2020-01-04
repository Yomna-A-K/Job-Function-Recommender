from flask import Flask, request, jsonify # loading in Flask
app = Flask(__name__)

from model_scripts.job_function_recommender import job_function_recommender

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # Make prediction
    title = str(data['title'])
    # making predictions
    preds = job_function_recommender(title)
    pred_str = ''
    index = 0
    for pred in preds:
        if index > 0:
            pred_str = pred_str +', '
        pred_str = pred_str + pred
        index = index + 1
    # returning the predictions as json
    return jsonify(pred_str)

if __name__ == '__main__':
    app.run(port=3000, debug=True)