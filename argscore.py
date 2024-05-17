from flask import Flask, request, render_template
import predict_arg_score as pred

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html', input = "", score = "")

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    answer = pred.get_predicted_arg_score(text.lower())
    # processed_text = str(answer)
    return render_template('form.html', input = "Your argument, \"" + text + "\", received a score of", score = str(answer))
