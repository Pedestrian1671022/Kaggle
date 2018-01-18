import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
train=pd.read_csv("train.csv")
train_text=train.text.values
train_author=train.author.values
name_vector={'EAP':0,'HPL':1,'MWS':2}

train_label=[name_vector[name] for name in train_author]

train_text_list=list(train_text)
lemm=WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer=super(LemmaCountVectorizer,self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

tf_vectorizer=LemmaCountVectorizer(max_df=0.95,min_df=2,stop_words='english',decode_error='ignore')
tf_bow=tf_vectorizer.fit_transform(train_text_list)

from sklearn.cross_validation import train_test_split
from sklearn import svm
train_x,test_x,train_y,test_y=train_test_split(tf_bow,train_label)
clf=svm.SVC()
clf.fit(train_x,train_y)
print clf.score(train_x,train_y)
print clf.score(test_x,test_y)

lda=LatentDirichletAllocation(max_iter=5,learning_method='online',
                              learning_offset=50.,random_state=0)
lda.fit(tf_bow)
n_top_words=30
print("\nTopics in LDA model:")
tf_feature_names=tf_vectorizer.get_feature_names()
# print_top_words(lda,tf_feature_names,n_top_words)
first_topic=lda.components_[0]
second_topic=lda.components_[1]
third_topic=lda.components_[2]
fourth_topic=lda.components_[3]
print first_topic
print second_topic
# print train_text_list
