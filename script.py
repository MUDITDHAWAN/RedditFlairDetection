#importing libraries
import os
# import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

PORT = int(os.environ.get("PORT",5000))

df = pd.read_csv("datawithoutnanflair.csv")

vectorizer = TfidfVectorizer()
x3=df.url
yo=df.flair
x8=df.permalink


i=0
y=[]
xnew=[]
# for x in x3:
#     y.append(yo[i])
#     i+=1
for x in x8:
    if(yo[i] in ['Politics','Policy/Economy','Science/Technology',"Non-Political","AskIndia",'[R]eddiquette','Business/Finance',"Food",'Photography',"Sports"]):
        y.append(yo[i])
        j=x.split("/")
        j=j[-2].replace("_"," ")
        xnew.append(j)
    i+=1


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
Y= label_encoder.fit_transform(y)

x8=vectorizer.fit_transform(xnew)

# prediction function
def ValuePredictor(to_predict_list):
    # to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("model3.pkl","rb"))
    result = loaded_model.predict(to_predict_list)
    result=label_encoder.inverse_transform(result)
    return result

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
@app.route('/index',methods = ['POST'])
def index():
    return flask.render_template('index.html')


@app.route('/statistics',methods = ['POST'])
def statistics():
    return flask.render_template('statistics.html')


@app.route('/result',methods = ['POST'])
def result():

    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        print(to_predict_list)
        to_predict_list=to_predict_list[0].split("/")[-2].replace("_"," ")
        print(to_predict_list)
        new=[]
        new.append(to_predict_list)
        to_predict=vectorizer.transform(new)


        # to_predict_list = list(map(float, to_predict_list))
        # print(to_predict_list)
        result = ValuePredictor(to_predict)
        prediction=result

        # if int(result)==1:
        #     prediction='Income more than 50K'
        # else:
        #     prediction='Income less that 50K'

        return render_template("result.html",prediction=prediction)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = PORT, debug=True)
