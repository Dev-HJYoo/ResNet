import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras import Model
from resnet import Resnet

if __name__ == "__main__":
	# Test Parameter
	Decay = 0.0001
	Batch_size = 64
	model_size = 5
	save_dir = './log/Data_Augmentation/ResNet32_batch64'

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
	model.load_weights(save_dir, Decay)

	# Evaluate Model with testset
	loss, acc = model.evaluate(test_dataset)
	print("Loss : {:0.4f}\nAccuracy: {:0.4f}".format(loss, acc))
