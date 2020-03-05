import tensorflow as tf
import numpy as np
import datetime as dt

from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D,\
MaxPooling2D, ZeroPadding2D, Add, ZeroPadding3D
from tensorflow.keras import Model


## solve
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
## 
def dataset():
	# Load CIFAR-10 
	(x_train, y_train), (x_test, y_test) = load_data()

	# # one_hot_encoding
	# y_train = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
	# y_test = tf.squeeze(tf.one_hot(y_test, 10), axis=1)
	y_train = tf.one_hot(y_train, 10)
	y_test = tf.one_hot(y_test, 10)

	# 0~ 255 -> 0~1
	x_train = x_train / 255
	y_train = y_train / 255
	x_test = x_test / 255
	y_test = y_test / 255
	x_val = x_train[-10000:]
	y_val = y_train[-10000:]
	return x_train, y_train, x_test, y_test, x_val, y_val
	

class Resnet():

	def __init__(self, number, name):
		self.name = name
		self.number = number
		self.m = tf.keras.Input(shape=(32,32,3))
		self._build_net()

	def _build_net(self):
		m = self.m
		
		# # First conv layer
		inputs = m
		m = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(m)
		m = BatchNormalization()(m)
		m = Activation('relu')(m)
		outputs = m

		#print('First conv layer')
		self.prints(inputs,outputs,'First_conv')
		

		# First Residual Block
		#print('First Residual Block\n\n')
		m, outputs = self.residual_block(m, self.number, 16, True)
		# self.prints(inputs,outputs,'First_residual_block')

		# Second Residual Block
		m, outputs = self.residual_block(m, self.number, 32)
		self.prints(inputs,outputs,'Second_residual_block')

		# Third Residual Block
		m, outputs = self.residual_block(m, self.number, 64)
		#self.prints(inputs,outputs,'Third_residual_block')

		# Average pool
		m = GlobalAveragePooling2D()(m)

		# 10-fc
		m = Dense(10, activation='softmax')(m)
		m = tf.expand_dims(m,axis=2)
		self.prints(inputs, m, 'Full')

		self.m = m
		self.model = Model(inputs,self.m,name='Resnet_'+str(2*self.number*3 + 2))

		return self.model

	def prints(self, inputs, outputs, name):
		Model(inputs=inputs, outputs=outputs, name=name).summary()

	def residual_block(self, m, n, filter, first=False):
	
		if not first:
			m = MaxPooling2D((1,1),2)(m)

		for i in range(n):
	
			shortcut = m
			if not first and i == 0:
				shortcut = Conv2D(filters=filter, kernel_size=(1,1), padding='same')(shortcut)
				shortcut = BatchNormalization()(shortcut)
				print(shortcut)
			m = Conv2D(filters=filter, kernel_size=(3,3), padding='same')(m)
			m = BatchNormalization()(m)
			m = Activation('relu')(m)
			m = Conv2D(filters=filter, kernel_size=(3,3), padding='same')(m)
			m = BatchNormalization()(m)
			m = Add()([m,shortcut])
			m = Activation('relu')(m)

		outputs = m

		return m, outputs

	def compile(self):
		return self.model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',
			metrics=['acc'])

	def fit(self, x, y, epoch, batch_size, validation_data, val_step, callbacks):
		return self.model.fit(x, y, epochs=epoch, batch_size=batch_size, 
			validation_data=validation_data, validation_steps=val_step, callbacks=callbacks)




x_train, y_train, x_test, y_test, x_val, y_val = dataset()
print(x_train.shape)
print(y_train.shape)
callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
]
model = Resnet(3, 'resnet')

compiles = model.compile()

fit = model.fit(x_train, y_train, 3, 128, [x_val,y_val], 3, callbacks)


