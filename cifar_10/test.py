import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras import Model
from resnet import Resnet

if __name__ == "__main__":
	# Test Parameter
	Decay = 0.0001
	momentum = 0.9
	Batch_size = 64
	model_size = 9
	save_dir = './log/ResNet' + str(model_size*6+2) + '_batch' + str(Batch_size) + '/'
	

	# Test Data Load
	(x_train, y_train), (x_test, y_test) = load_data()

	# one_hot_encoding
	y_test = tf.one_hot(y_test, 10)

	# Data nomarization
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(Batch_size)
	test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255, y))

	# Model Generate
	model = Resnet(model_size, 'Resnet')

	# Load weights
	model.load_weights(save_dir, momentum, Decay)

	# Evaluate Model with testset
	loss, acc = model.evaluate(test_dataset)

	# Save test result with tensorboard
	writer = tf.summary.create_file_writer(save_dir+'tests')
	with writer.as_default():
		tf.summary.scalar('Test_Accuracy', data=acc, step=0)
		tf.summary.scalar('Test_Loss', data=loss, step=0)

	print("Loss : {:0.4f}\nAccuracy: {:0.4f}".format(loss, acc))