import numpy as np
import jieba
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import sys

np.random.seed(1337)  # for reproducibility

print ('Loading data.....')
filename = 'input.txt'
data = open(filename, 'r').read()  # should be simple plain text file
data = data.decode('utf-8')
data = list(jieba.cut(data, cut_all=False))
chars = list(set(data)) # 所有不重复的词的列表
data_size, vocab_size = len(data), len(chars)



print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }



# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, data_size-seq_length, 1):
    seq_in = data[i:i+seq_length]
    seq_out = data[i+seq_length]
    dataX.append([char_to_ix[char] for char in seq_in])
    dataY.append(char_to_ix[seq_out])
n_patterns = len(dataX)

print ("Total Patterns:", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
y = np.reshape(dataX, (n_patterns, seq_length))

print ("X's shape:", X.shape)
print ("y's shape:", y.shape)


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, nb_epoch=800, batch_size=128, verbose=1)


# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_size)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = ix_to_char[index]
    seq_in = [ix_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone!")

