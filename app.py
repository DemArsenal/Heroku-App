import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

app = Flask('mymarketvaluepredictorApp')

# route 1: hello world
@app.route('/')
def home():
    # return a simple string
    return 'Welcome to my app, this is meant to help predict player market value. Enjoy!'


# route 2: show a form to the user
@app.route('/form')
def form():
    # use flask's render_template function to display an html page
    return render_template('form.html')


# route 5: accept the form submission and do something fancy with it
@app.route('/submit')
def make_predictions():
    # load in the form data from the incoming request
    user_input = request.args

    # manipulate data into a format that we pass to our model
    data = np.array([
        int(user_input['wage_eur']),
        int(user_input['overall']),
        int(user_input['potential']),
        int(user_input['skill_moves']),
        int(user_input['weak_foot'])
    ]).reshape(1,-1)

    model = pickle.load(open('model.p','rb'))

    prediction = model.predict(data)[0]

    return render_template('results.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
