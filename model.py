import re
import utils
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

data = pd.read_csv('data/Sentiment.csv')

# Keeping necessary data
data = data[['text', 'sentiment']]

data['text'] = data['text'].apply(utils.preprocess_data)

# Tokenizing data and padding sentences
max_features = 2000

tokenizer = Tokenizer(num_words = max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, 28)

Y = pd.get_dummies(data['sentiment']).values

# Splitting into training and testing portions
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

# Embedding layer and LSTM with dropout
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model
batch_size = 512
model.fit(x_train, y_train, epochs = 10, batch_size=batch_size, validation_data=(x_test, y_test))

# Saving the model
model.save('sentiment.h5')
