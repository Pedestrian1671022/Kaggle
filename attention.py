import tensorflow as tf

def attention(inputs,attention_size,time_major=False,return_alphas=False,regularization=False):
    if isinstance(inputs,tuple):
        #In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs=tf.concat(inputs,2)
    if time_major:
        inputs=tf.array_ops.transpose(inputs,[1,0,2])

    inputs_shape=inputs.shape
    # sequence_length=inputs_shape[1].value
    hidden_size=inputs_shape[2].value

    W_omega=tf.Variable(tf.random_normal([1,hidden_size,attention_size],stddev=0.1))
    b_omega=tf.Variable(tf.random_normal([attention_size],stddev=0.1))
    u_omega=tf.Variable(tf.random_normal([1,attention_size,1],stddev=0.1))
    if regularization:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(W_omega))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.nn.l2_loss(b_omega))
    v=tf.tanh(tf.matmul(inputs,tf.tile(W_omega,tf.stack([tf.shape(inputs)[0],1,1])))+b_omega)
    vu=tf.matmul(v,tf.tile(u_omega,tf.stack([tf.shape(inputs)[0],1,1])))
    exps=tf.reshape(tf.exp(vu),[tf.shape(inputs)[0],-1])
    alphas=exps/tf.reshape(tf.reduce_sum(exps,1),[-1,1])
    output=tf.reduce_sum(inputs*tf.reshape(alphas,[-1,tf.shape(inputs)[1],1]),1)

    if not return_alphas:
        return output
    else:
        return output,alphas