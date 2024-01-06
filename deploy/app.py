from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

model = joblib.load('./model.joblib')
vocabulary = joblib.load('./vocabulary.joblib')
classes = {0: 'Facebook', 1: 'Twitter'}
cvec = CountVectorizer(max_features = 500, stop_words = 'english', vocabulary = vocabulary)

@app.route('/', methods = ["GET", "POST"])
def predict():
    output = None
    postText = None
    if request.method == "POST":
        postText = str(request.form['postText'])
        input_cvec = pd.DataFrame(cvec.transform([postText]).todense(), columns = cvec.get_feature_names())
        output = classes[model.predict(input_cvec)[0]]
    return render_template(
      "predict.html",
      output = output,
      postText = postText
      )

if __name__ == '__main__':
  app.run()