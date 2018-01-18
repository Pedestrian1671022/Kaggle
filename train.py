import tensorflow as tf
import numpy as np
from attention import attention
from data_process import next_batch

input_sequence=tf.placeholder(tf.float32,[None,None,50])#batch_size,squence_size,vecoter_size
input_label=tf.placeholder(tf.float32,[None,3])
seq_len_ph=tf.placeholder(tf.int32,[None])
keep_prob_ph=tf.placeholder(tf.float32)
'''
parameters
'''
batch_size=64
hidden_units=128
hidden_layer=3
attention_size=50
keep_prob=0.8
lr=1e-3

gru_fw_cell=tf.contrib.rnn.GRUCell(hidden_units)
gru_bw_cell=tf.contrib.rnn.GRUCell(hidden_units)

rnn_outputs,_=tf.nn.bidirectional_dynamic_rnn(gru_fw_cell,gru_bw_cell,input_sequence,sequence_length=seq_len_ph,dtype=tf.float32)
attention_output=attention(rnn_outputs,attention_size,return_alphas=False)

drop=tf.nn.dropout(attention_output,keep_prob_ph)

w=tf.Variable(tf.truncated_normal([drop.get_shape()[1].value,3],stddev=0.1))
b=tf.Variable(tf.constant(0.,shape=[3]))
y_hat=tf.nn.xw_plus_b(drop,w,b)
y_hat=tf.squeeze(y_hat)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_label,logits=y_hat))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.softmax(y_hat)),input_label),tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(3):
        for i in range(1000):
            x_batch,y_batch,seq_len=next_batch(batch_size).next()
            train_loss,acc,_=sess.run([loss,accuracy,optimizer],feed_dict={input_sequence:x_batch,input_label:y_batch,
                                                                     seq_len_ph:seq_len,keep_prob_ph:1.0})
            if i%50==0:
               print('number epoch %d,number step %d loss is %f'%(epoch,i,train_loss))
               print('number epoch %d,number step %d accuracy is %f'%(epoch,i,acc))
    for j in range(100):
        x_batch,y_batch,seq_len=next_batch(batch_size,train=False).next()
        acc=sess.run(accuracy,feed_dict={input_sequence:x_batch,input_label:y_batch,seq_len_ph:seq_len,keep_prob_ph:1})
        if j%50==0:
           print('number step %d accuracy is %f'%(j,acc))
        accuracy_train/=1000
        print accuracy_train
