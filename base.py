import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

x = np.array(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
y = np.array(5,6,28,30,31,34,39,44,50,60,73,81,97,107,118,137,151,173,223,283,360,434,519,606,694,834,918,1024,1251,1397,1834,2069,2547,3072,3374)

n = len(x)

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(),name ="W")
B = tf.Variable(np.random.randn(),name ="B")

learning_rate = 0.01
training_epochs = 1000

y_pred = tf.add(tf.multiply(X,W),B)
cost = tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)
optimizer = tf.train.GradientDescentOptimizer(learning_rate.minimize(cost))
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(training_epochs):
    for (_x,_y) in zip(x,y):
      sess.run(optimizer, feed_dict = {X: _x,Y: _y})
    if (epoch + 1)%50 == 0:
      c = sess.run(cost,feed_dict = {X: x,Y: y})
      print("Epoch", (epoch+1),": cost =",c,"W =",sess.run(w), "B =", sess.run(B))
  training_cost = sess.run(cost,feed_dict = {X:x,Y:y})
  weight = sess.run(W)
  bias = sess.run(B)

predictions = weight * x + bias
print("Training cost =", training_cost, "Weight =",weight,"bias =",bias,'\n')

plt.plot(x, y, 'ro', label = "Initial Data")
plt.plot(x ,predictions, label ="Output Line")
plt.title('COVID 19 Prediction Model Output')
plt.legend()
plt.show()
