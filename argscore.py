from flask import Flask, request, render_template
import predict_arg_score as pred
import get_advice as adv

def clean_up_text(txt):
    return txt.lower()

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html', input = "", score = "", advice = "")

@app.route('/', methods=['POST'])
def my_form_post():
    raw_text = request.form['text']
    text = clean_up_text(raw_text)
    score_float = pred.get_predicted_arg_score(text)
    return render_template('form.html', 
                           input = "Your argument, \"" + raw_text + "\", received a score of",
                           score = str(score_float),
                           advice = adv.get_advice_f(text, score_float))
