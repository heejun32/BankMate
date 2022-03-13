# CNN 기반 답변 처리 함수

from konlpy.tag import Kkma
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import csv
import tensorflow as tf


model = tf.keras.models.load_model('E:/Github/Bankmate/static/data/bankmate_cnn_model_v2.h5')

class CnnBased():
    def __init__(self, question):
        self.question = question
        self.train_data = self.call_train_data()
        self.tokenized_question = self.tokenizer_question(self.question)
        self.test_input = self.make_question(self.tokenized_question)
        self.intent, self.test_predictions_probas, self.test_predictions = self.make_intent()

    def call_train_data(self):
        tokenized_expand_data = list()
        f = open("E:/Github/Bankmate/static/data/save.csv", 'r',
                 encoding='utf-8')  # 경로 다시 설정해 주세요!
        rea = csv.reader(f)
        for row in rea:
            while '' in row:
                row.remove('')  # '' 삭제
            tokenized_expand_data.append(row)
        f.close

        train_data = []
        for i in range(0, len(tokenized_expand_data)):
            train_data.append(' '.join(tokenized_expand_data[i]))

        return train_data

    def tokenizer_question(self, question):
        specialChars = "!#$%^&*()?,."
        for specialChar in specialChars:
            self.question = self.question.replace(specialChar, '')
        kkma = Kkma()
        tokenized_question = []
        tokenized_sentence = kkma.morphs(self.question)  # 토큰화
        tokenized_question.append(tokenized_sentence)
        return tokenized_question

    def make_question(self, tokenized_question):
        input_question = []
        max_len = 17
        for i in range(0, len(tokenized_question)):
            input_question.append(' '.join(tokenized_question[i]))

        tokenizer = Tokenizer(oov_token="OOV")
        tokenizer.fit_on_texts(self.train_data)
        test_sequences = tokenizer.texts_to_sequences(input_question)
        test_input = pad_sequences(test_sequences, maxlen=max_len)
        return test_input

    def make_intent(self):
        label_expand_data = pd.read_csv(
            'E:/Github/Bankmate/static/data/save2.csv', encoding="UTF-8",
            header=None)  # 경로 다시 설정해 주세요!
        # Get predictions
        test_predictions_probas = model.predict(self.test_input)
        test_predictions = test_predictions_probas.argmax(axis=-1)

        idx_encode = preprocessing.LabelEncoder()
        idx_encode.fit(label_expand_data)

        test_intent_predictions = idx_encode.inverse_transform(test_predictions)
        intent = test_intent_predictions[0]
        return intent, test_predictions_probas, test_predictions

    def print_answer(self):
        # 답변 출력

        if self.test_predictions_probas[:, self.test_predictions[0]][0] > 0.99:
            df = pd.read_csv(
                "E:/Github/Bankmate/static/data/bankmate_answer.csv",
                encoding="utf-8")  # 경로 다시 설정해 주세요!

            pd.set_option('display.max_colwidth', None) # 답변 뒷부분 생략 없이 다 뜨게 함
            #name, dtype 없애고 줄바꿈 반영해서 리턴값에 반영
            str1 = pd.Series.to_string(df[df['intent'] == self.intent]['answer'], name=False, dtype=False)
            str2 = str1.replace("\n","<br>").replace("\\n", "<br>")
            return str2
        else:
            return "답변을 찾을 수 없습니다. 질문을 다시 확인해 주세요."