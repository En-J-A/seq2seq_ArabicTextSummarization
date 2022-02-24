import pickle

import flask
import numpy as np
import numpy as np
import pickle


from tensorflow import keras
from flask import Flask, render_template, url_for,request, jsonify
from keras.preprocessing.sequence import pad_sequences
from utils.utils import  *
import logging

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)




####loading models

seq2seq_model = keras.models.load_model('seq2seq_50epoch.h5')
encoder_model = keras.models.load_model('encoder_model.h5')
decoder_model = keras.models.load_model('decoder_model.h5')



## loading tokenizers

with open('xtokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)

with open('ytokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

## constants
max_text_len=500
max_summary_len=13



app = Flask(__name__)




@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():


    if request.method == 'POST':
        logging.info(request)
        text = request.form['Article']
        # text2 = request.form.get('data')
        # search = request.get_json()
        logging.info(request)
        # logging.info("reesponse text:")
        # logging.info(text)
        # logging.info(text2)
        # logging.info(search)


        logging.info("reesponse text type:")
        # text = 'كد ياسر عثمان سفير مصر في المغرب، أن مباراة نادي الزمالك والوداد المغربي في دوري أبطال أفريقيا، ستقام بدون حضور جماهيري.'
        logging.info(type(text))

        preprocessed = preprocessing(text)
        test_seq = x_tokenizer.texts_to_sequences([preprocessed])
        final_test_seq = pad_sequences(test_seq, maxlen=max_text_len, padding='post')
        final_test_seq = final_test_seq[0]
        # print("Review:", seq2text(final_test_seq))
        # print("Original summary:",seq2summary(y_tr[i]))
        predicted = decode_sequence(final_test_seq.reshape(1, max_text_len),encoder_model,decoder_model,target_word_index,reverse_target_word_index,max_summary_len)
        logging.info('predicted:')
        logging.info(predicted)

        # print("Predicted summary:",)
        # print("\n")
        # return predicted
        predicted = predicted[7:-4]
        return render_template('pred.html',data= predicted)


# @app.route('/api/v1/predict/<string:text>', methods=['GET'])
# def predict_api(text):
#
#     JobTitle = text
#     JobTitle = np.array([JobTitle])
#
#     sample = vectorizer.transform(JobTitle).toarray()
#
#     preds = model.predict(sample)
#     prediction = preds[0]
#
#     response = {
#         "text": text,
#         "prediction": prediction
#     }
#     return jsonify(response)


if __name__ == '__main__':
    app.run(port=8888,   debug=True)
