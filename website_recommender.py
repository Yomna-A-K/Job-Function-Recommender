from flask import Flask, request, render_template

import pandas as pd

app = Flask(__name__, template_folder="website_template")

from model_scripts.job_function_recommender import job_function_recommender
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form.get('title')
        # Make prediction
        pred = job_function_recommender(title)
        pred_str = ''
        i=0
        for function in pred:
            if i > 0:
                pred_str = pred_str + ', '
            pred_str = pred_str + function
            i = i+1
        return render_template('index.html', recommended_functions=pred_str)
    return render_template('index.html', recommended_functions='')


if __name__ == '__main__':
    app.run(port=3000, debug=True)