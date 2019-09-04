from flask import Flask, request, render_template
import pickle
from sklearn.externals import joblib

with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)
#joblib_file = "joblib_model.pkl"
#classifier = joblib.load(joblib_file)


def get_predictions(*args):
    mylist = [*args]
    mylist = [float(i) for i in mylist]
    #print(mylist)
    vals = [mylist]
    #print(vals)
    return classifier.predict(vals)[0]


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    cainput = request.form['cainput']
    thal = request.form['thal']

    target = get_predictions(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope,cainput, thal)
    #return str(target)
    if target == 1:
        text_out ="Person is likely to get Heart Disease"
    else:
        text_out = "Person is unlikely to get Heart Disease"
    return render_template('home.html', target = target, text_out =text_out)


if __name__ == "__main__":
    app.run(debug=True)