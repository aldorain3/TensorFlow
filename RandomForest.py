# Random Forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
num_steps = 500
batch_size = 538
num_classes = 2
num_features = 95
num_trees = 10
max_nodes = 1000
tf.reset_default_graph() 
X = tf.placeholder(tf.float32,[None, 95])

Y = tf.placeholder(tf.float32,[None,2])

hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

forest_graph = tensor_forest.RandomForestGraphs(hparams)

train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)


infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 0), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

sess = tf.Session()

sess.run(init_vars)

for i in range(1, num_steps + 1):

    l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})

