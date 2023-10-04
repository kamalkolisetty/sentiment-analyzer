from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the tokenizer and model outside of the process_input function
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
@app.route('/process', methods=['POST'])

def process():
    if request.method == 'POST':
        input_text = request.form['input_text']
        processed_output = process_input(input_text)
        return jsonify(processed_output) 

def process_input(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2])
    }

    # Determine sentiment based on the highest score
    if scores_dict['roberta_neg'] > scores_dict['roberta_neu'] and scores_dict['roberta_neg'] > scores_dict['roberta_pos']:
        sentiment = " NEGATIVE.\n Stay strong, my friend. You've got this! \U0001F4AA"

    elif scores_dict['roberta_pos'] > scores_dict['roberta_neu'] and scores_dict['roberta_pos'] > scores_dict['roberta_neg']:
        sentiment = " POSITIVE.\n Your positivity is contagious, and it brightens my day too! \U0001F604\U0001F31E"

    else:
        sentiment = " NEUTRAL.\n Happy day, mate! \U0001F31E\U0001F604"

    scores_dict['sentiment'] = sentiment
    return scores_dict



if __name__ == '__main__':
    app.run(debug=True)
