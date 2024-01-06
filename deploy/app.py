from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

mnb_model = joblib.load('./mnb_model.joblib')
subreddit = {0: '/r/relationship_advice', 1: '/r/personalfinance'}

@app.route('/', methods = ["GET", "POST"])
def predict_interface():
    output = None
    proba = None
    postTitle = None
    postText = None

    if request.method == "POST":
        postTitle = str(request.form['postTitle'])
        postText = str(request.form['postText'])
        user_input = postTitle + ' ' + postText
        output = subreddit[mnb_model.predict([user_input])[0]]
        mnb_prob = mnb_model.predict_proba([user_input]).tolist()
        proba_dict = {subreddit[k]: round(mnb_prob[0][i] * 100, 2) for i, k in enumerate(mnb_model.classes_.tolist())}
        proba = dict(sorted(proba_dict.items(), key=lambda x:x[1], reverse=True))
    return render_template(
      "predict.html",
      output = output,
      proba = proba,
      postTitle = postTitle,
      postText = postText
      )

if __name__ == '__main__':
  app.run()