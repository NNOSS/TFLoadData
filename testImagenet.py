import tensorflow as tf
import loadDataset
SUMMARY_FILEPATH = '/Models/ImageNet/Summaries/'

dataset = loadDataset.ImageNet.return_dataset_train().repeat().batch(30)
train_iterator = dataset.make_initializable_iterator()
train_input, train_label = train_iterator.get_next()
input_summary = tf.summary.image("image_inputs", train_input)
sess = tf.Session()
train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                  sess.graph)
i = 0
sess.run(train_iterator.initializer)
while True:
    instance = sess.run(input_summary)
    train_writer.add_summary(instance, i)
    i += 1
