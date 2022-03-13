from gensim.models import Word2Vec
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터 디렉토리 경로 설정
path = "E:\Github\Bankmate\static\data\\"

# 문장에서 랜덤으로 선택된 단어 n개에 대하여 유의어 대체
def synonym_replacement(words, n):
	new_words = words.copy()
	np.random.seed(1103)
	random_word_list = list(set([word for word in words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1
		if num_replaced >= n:
			break

	return new_words

def get_synonyms(word):
	synomyms = []

	try:
		for syn in synonym_dict[word]:
				synomyms.append(syn)
	except:
		pass

	return synomyms

# 유의어 사전 생성
synonym_dict = {}
with open(path+"synonym.csv",'r', encoding="UTF-8") as f:
    reader = csv.reader(f)
    next(reader)
    for lines in reader:
      while '' in lines:
	      lines.remove('') # '' 삭제
      synonym_dict[lines[0]] = lines[1:]


tokenized_expand_data=list()
f = open(path+"save.csv", 'r', encoding='utf-8')
rea = csv.reader(f)
for row in rea:
  while '' in row:
    row.remove('') # '' 삭제
  tokenized_expand_data.append(row)
f.close

label_expand_data=list()
f = open(path+"save2.csv", 'r', encoding='utf-8')
rea = csv.reader(f)
for row in rea:
  while '' in row:
    row.remove('') # '' 삭제
  label_expand_data.append(row)
f.close

def csv2list(filename):
  import csv
  file = open(filename, 'r', encoding="UTF-8")
  csvfile = csv.reader(file)
  lists = []
  for items in csvfile:
    lists.append(items)
  return lists

tokenized_data = csv2list(path+'kkma_tokenized_data.csv')
tokenized_nouns = csv2list(path+'kkma_tokenized_nouns.csv')


# 임베딩 단계
model_wv = Word2Vec(sentences = tokenized_expand_data, vector_size = 100, window = 2, min_count = 10, workers = 2, sg = 1, epochs=50)

# 1D CNN
train_data=[]
for i in range(0, len(tokenized_expand_data)):
  train_data.append(' '.join(tokenized_expand_data[i]))

label_data = pd.read_csv(path + 'bankmate_train2.csv', encoding="UTF-8")['label']
label_data = label_data.tolist()

# 레이블 인코딩. 레이블에 고유한 정수를 부여
idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(label_expand_data)
label_expand_data = idx_encode.transform(label_expand_data) # 주어진 고유한 정수로 변환
label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))

tokenizer = Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(train_data)
sequences = tokenizer.texts_to_sequences(train_data)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 2

max_len = 17
question_train = pad_sequences(sequences, maxlen = max_len)

indices = np.arange(question_train.shape[0])
np.random.seed(1103)
np.random.shuffle(indices)

question_train = question_train[indices]
label_train = label_expand_data[indices]

n_of_val = int(0.2 * question_train.shape[0])

# 훈련, 검증 데이터 분리
X_train = question_train[:-n_of_val]
y_train = label_train[:-n_of_val]
X_val = question_train[-n_of_val:]
y_val = label_train[-n_of_val:]

embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for i in range(len(model_wv.wv.vectors)):
    embedding_vector = model_wv.wv.vectors[i]
    if embedding_vector is not None:
        embedding_matrix[i+1] = embedding_vector

# 모델 구성
kernel_sizes = [1, 3, 3]
num_filters = 512
dropout_ratio = 0.5
np.random.seed(1103)

model_input = Input(shape=(max_len,))
output = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                      input_length=max_len, trainable=False)(model_input)

conv_blocks = []

for size in kernel_sizes:
    conv = Conv1D(filters=num_filters, kernel_size=size, padding="valid",
                  activation="relu", strides=1)(output)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

output = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
output = Dropout(dropout_ratio)(output)
model_output = Dense(len(label_train), activation='softmax')(output)
model = Model(model_input, model_output)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 구성 끝

# 모델 학습 및 결과 확인
history = model.fit(X_train, y_train,
          batch_size=64,
          epochs=10,
          validation_data=(X_val, y_val))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()