
#imports
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation
# from keras.utils import np_utils

#const
sps = 8000
bit_rate = 8
max_val = 2**bit_rate
note_size = 10 #int(sps/1000)
name_list = ['Breaking Benjamin w- The Diary of Jane.wav','Dua Lipa - New Rules (Official Music Video) (1).wav']

#fetch 30 sec from a song
print("start fetching")
from scipy.io import wavfile
dat=[]
for name in name_list:
	fs, d = wavfile.read(name)
	dat.append(d)
data=sum(dat)
frame_num = len(data)
#
print("# of frames,",frame_num)
#norm
print("normalizing")
def norm(x):
	return x[0]/max_val -0.5
	# return x[0]
norm_data = list(map(norm,data))
print ("norm_data")
for i in range(8000,8010):
	print(norm_data[i])


#encoder decoder data
print("reshape data")
inp = []
for i in range(int(frame_num/note_size)):
	inp.append(np.array([norm_data[i*note_size+j] for j in range(note_size)]))
exp = inp
print("inp[99] = ",inp[99])

train_test_rait = 0.6

X_train, y_train = np.array(inp[:int(len(inp)*train_test_rait)]),np.array(exp[:int(len(inp)*train_test_rait)])
X_test, y_test = np.array(inp[int(len(inp)*train_test_rait):]),np.array(exp[int(len(inp)*train_test_rait):])

# print(X_train.shape,X_train[0], X_train[0].shape)

#model
print("setting model")
model = Sequential()
model.add(Dense(units=9, activation='tanh', input_dim=note_size))
model.add(Dense(units=9, activation='tanh'))
# model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=note_size, activation='tanh'))

print("compiling")
model.compile(optimizer='rmsprop',loss='mse')

print("fitting")
model.fit(X_train, y_train, epochs=20, batch_size=32)

print("evaluating")
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
print(loss_and_metrics)
print(model)

print("let's rock")
rock =  model.predict(np.array(inp), batch_size=128)
for i in range(5):
	print(rock[10000+i],"->\n",inp[10000+i],"\n")
data2 = []

for i in range(int(frame_num/note_size)):
	data2 += list(rock[i])

def deNorm(x):
	x+=0.5
	return [int(x*max_val),int(x*max_val)]
	# return [x,x]
de_norm_data = np.array(list(map(deNorm,data2)))

for i in range(5):
	print(de_norm_data[10000+i*100],"->\n",data[10000+i*100],"\n")

# print("we will write:",de_norm_data)

print("de_norm_data:",de_norm_data,"type",type(de_norm_data))
print("data:",data,"type",type(data))

for i in range(len(data)):
	data[i] = de_norm_data[i]

print("data2:",data,"type",type(data))

wavfile.write('bb-lets rock7.wav', sps, de_norm_data)
wavfile.write('mix.wav', sps, data)