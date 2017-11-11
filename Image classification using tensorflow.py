
#Import MNIST dataset
def import_mnist_dataset():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
    return mnist

if __name__ == '__main__':

    #Library to be used
    import tensorflow as tf
    import numpy as np

    # File to save the trained model
    save_file='./model.ckpt'


    # Load data set
    mnist=import_mnist_dataset()

    # print some insformation about the data set
    print("The dimension of the training set is ",mnist.train.images.shape)
    print("The dimension of the test set is",mnist.test.images.shape)
    print("The dimension of the validation set is ",mnist.validation.images.shape)

    # Extracting features and labels
    train_imgs=mnist.train.images
    train_labels=mnist.train.labels

    test_imgs=mnist.test.images
    test_labels=mnist.test.labels

    validation_imgs=mnist.validation.images
    validation_labels=mnist.validation.labels

    img_vector_width=train_imgs.shape[1]
    label_vector_width=train_labels.shape[1]

    # Network Architecture as a computational graph
    # Input features
    x=tf.placeholder(tf.float32,[None,img_vector_width])
    # Labels
    y=tf.placeholder(tf.float32,[None,label_vector_width])

    # Dropout portion
    keep_prob=tf.placeholder(tf.float32)

    # Hidden layers: two hidden layers with 512 neurons each with 'ReLU' activation
    # First hidden layer : img_vector_width x 512
    num_neur_hidden_1=512
    w_hidden_1=tf.Variable(tf.random_normal([img_vector_width,num_neur_hidden_1]))
    b_hidden_1=tf.Variable(tf.random_normal([num_neur_hidden_1]))
    output_hidden_1=tf.add(tf.matmul(x,w_hidden_1),b_hidden_1)
    output_hidden_1=tf.nn.relu(output_hidden_1)
    output_hidden_1=tf.nn.dropout(output_hidden_1,keep_prob)
    # Second hidden layer
    num_neur_hidden_2=512
    w_hidden_2=tf.Variable(tf.random_normal([num_neur_hidden_1,num_neur_hidden_2]))
    b_hidden_2=tf.Variable(tf.random_normal([num_neur_hidden_2]))
    output_hidden_2=tf.add(tf.matmul(output_hidden_1,w_hidden_2),b_hidden_2)
    output_hidden_2=tf.nn.relu(output_hidden_2)
    output_hidden_2=tf.nn.dropout(output_hidden_2,keep_prob)

    #ouptput layer with softmax activation
    w_hidden_ouput=tf.Variable(tf.random_normal([num_neur_hidden_2,label_vector_width]))
    b_hidden_ouput=tf.Variable(tf.random_normal([label_vector_width]))
    logits=tf.add(tf.matmul(output_hidden_2,w_hidden_ouput),b_hidden_ouput)
    #output=tf.nn.softmax(output)

    saver=tf.train.Saver()

    # Hyper parameters from training
    learning_rate=0.01
    batch_size=256
    epochs=100

    # Cost function and optimizer selection
    # Cost function is a crossentropy loss function
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
    #Optimizer Gradient descent
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Prediction and accuracy
    correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # Initialize the variables
    init=tf.global_variables_initializer()


    #Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Loop over all the epochs
        for epoch in range(epochs):
            total_batch=int(train_imgs.shape[0]/batch_size)
            # Loop over all the batches
            for i in range(total_batch):
                batch_x, batch_y=mnist.train.next_batch(batch_size)
                # Run the optimizer
                sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
        saver.save(sess,save_file)

        # Load the model and test it using validation set
        saver.restore(sess,save_file)
        test_accuracy= sess.run(accuracy,feed_dict={x:validation_imgs,y:validation_labels,keep_prob:1})
        print('Test accuracy is ',test_accuracy)
