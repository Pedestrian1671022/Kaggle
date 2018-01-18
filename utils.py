import csv
import nltk
import nltk.data
import numpy as np
from sklearn.cross_validation import train_test_split

train_data=csv.reader(open('train.csv'))
# test_data=csv.reader(open('test.csv'))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 

train_x=[]
train_y=[]
# test_x=[]
# test_y=[]
for line in train_data:
    train_x.append(line[1])
    train_y.append(line[2])
del train_y[0]
del train_x[0]

# for line in test_data:
#     test_x.append(line[1])
#     test_y.append(line[2])
# del test_y[0]
# del test_x[0]


name_vector={'EAP':0,'HPL':1,'MWS':2}
def onehot(x):
    x_onehot=np.zeros(3)
    x_onehot[x]=1
    return x_onehot
train_label=[]

for i in xrange(len(train_y)):
    train_label.append(tuple(onehot(name_vector[train_y[i]])))

'''
word to vector
'''

glove_wordmap={}
with open('glove.6B.50d.txt','r') as glove:
    for line in glove:
        name,vector=tuple(line.split(" ",1))
        glove_wordmap[name]=np.fromstring(vector,sep=" ")
wvecs=[]
for item in glove_wordmap.items():
    wvecs.append(item[1])
s=np.vstack(wvecs)

v=np.var(s,0)
m=np.mean(s,0)
RS=np.random.RandomState()

def fill_unk(unk):
    global glove_wordmap
    glove_wordmap[unk]=RS.multivariate_normal(m,np.diag(v))
    return glove_wordmap[unk]

def sentence2sequence(sentence):
    tokens=sentence.strip('"(),-').lower().split(" ")
    rows=[]
    words=[]
    for token in tokens:
        i=len(token)
        while len(token)>0:
            word=token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token=token[i:]
                i=len(token)
                continue
            else:
                i=i-1
            if i==0:
                rows.append(fill_unk(token))
                words.append(token)
                break
    return np.array(rows),words

def contextualize(data_set):
    data=[]
    context=[]
    for line in data_set:
        data.append(tuple(sentence2sequence(line)[0]))
        context.append(tuple(sentence2sequence(line)[1]))
    return data,context

train_vector,train_word=contextualize(train_x)
x_train,x_test,y_train,y_test=train_test_split(train_vector,train_label,test_size=0.2)
def next_batch(batch_size,train=True):
    if train:
        size=len(x_train)
        x_copy=np.array(x_train).copy()
        y_copy=np.array(y_train).copy()
    else:
        size=len(x_test)
        x_copy=np.array(x_test).copy()
        y_copy=np.array(y_test).copy()
    indices=np.arange(size)
    np.random.shuffle(indices)
    x_copy=x_copy[indices]
    y_copy=y_copy[indices]
    i=0
    while True:
        if i+batch_size<=size:
            x_batch=x_copy[i:i+batch_size]
            y_batch=y_copy[i:i+batch_size]
            seq_len = np.array([len(x) for x in x_batch])
            length=max(map(len,x_batch))
            x_data=np.zeros([batch_size,length,50])
            for row in range(batch_size):
                x_data[row, :len(x_batch[row])] = x_batch[row]
            yield x_data,y_batch,seq_len
            i+=batch_size
        else:
            i=0
            indices=np.arange(size)
            np.random.shuffle(indices)
            x_copy=x_copy[indices]
            y_copy=y_copy[indices]
            continue
# print next_batch(2).next()

# test_vector,test_word=contextualize(test_x)
