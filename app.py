import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

#procfile.txt
#web: gunicorn app:app
#first file that we have to run first : flask server name
app = Flask(__name__)
import joblib
new_model=joblib.load('finalmodel')




@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    data = pd.read_csv("ticket_cost.csv")
    inputs2 = data[["Country_code", "Class"]]
    target2 = data[["Ticket_price"]]
    model1 = DecisionTreeRegressor()
    model1.fit(inputs2, target2)
    new_vector = np.zeros(151)

    if request.method == 'POST':
        result = request.form
        new_vector[0] = result['Country_code']
        new_vector[1] = result['Days']
        new_vector[2] = result['Class']



    prediction1 = new_model.predict([[new_vector[0],new_vector[1]]])
    prediction1=np.round(prediction1)
    prediction2 = model1.predict([[new_vector[0],new_vector[2]]])
    #print(prediction)
    abc=""

    return render_template('index.html', Predict_score ='Cost of living is ₹ {} '.format(prediction1),abc="  ",Predict_score1 ='   and Airline ticket price is  ₹ {} '.format(prediction2))

if __name__ == "__main__":
    app.run(debug=True)
