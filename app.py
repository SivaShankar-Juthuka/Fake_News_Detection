import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv(r"C:/Users/Siva Shankar/Desktop/Fake News Detection/news.csv")

x = data['text']
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

def fake_news_data(news):
  tfid_x_train = tfvect.fit_transform(x_train)
  tfid_x_test = tfvect.transform(x_test)
  input_data = [news]
  vec_input_data = tfvect.transform(input_data)
  prediction = loaded_model.predict(vec_input_data)
  return prediction

@app.route('/')
def home():
  return render_template("index.html")

@app.route("/result", methods=["POST"])
def predict():
  if request.method == 'POST':
    message = request.form['message']
    pred = fake_news_data(message)
    print(pred)
    return render_template('index.html', prediction = pred)
  return None

if __name__ == '__main__':
  app.run(debug = True)