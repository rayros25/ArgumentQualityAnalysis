from flask import Flask, request, render_template
import predict_arg_score as pred
import get_advice as adv

# Preprocessing before handing off to the BERT model
def clean_up_text(txt):
    return txt.lower()

# The next two definitions are just asynchronous wrapper functions
async def get_score(txt):
    return pred.get_predicted_arg_score(txt)

async def get_adv(txt, scr):
    return adv.get_advice_f(txt, scr)

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html', input = "", score = "", advice = "")

@app.route('/', methods=['POST'])
async def my_form_post():
    raw_text = request.form['text']
    text     = clean_up_text(raw_text)
    score_float = await get_score(text)
    advice      = await get_adv(text, score_float)
    return render_template('form.html', 
                           input = "Your argument, \"" + raw_text +
                                   "\", received a score of",
                           score = str(score_float),
                           advice = advice)
