from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from transformers import BertTokenizerFast

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('path_to_your_model')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        input_encodings = tokenizer(
            [text], truncation=True, padding=True, max_length=512)
        predictions = model.predict(input_encodings['input_ids'])
        # Convert predictions to your desired format and add to the context
        context = {'predictions': predictions}
        return render_template('index.html', **context)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
