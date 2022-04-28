import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import zipfile
import shutil
import random
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pylab import rcParams
import sys

rcParams['figure.figsize'] = 20, 20

def read_data(filename):
	#Read dataset info
	with open(filename,'r') as f:
	  data=f.readlines()
	for index,item in enumerate(data):
	  if '\n' in item:
	    data[index]=item[:-1]
	return data

def write_to_csv(filename, data):
	with open(filename,'w',newline='') as f:
	  csvw=csv.writer(f)
	  csvw.writerow(['filename','class'])
	  for item in data:
	    class_name=item[:item.index('/')]
	    img_name='indoorCVPR_09/Images/'+item
	    csvw.writerow([img_name,class_name])
def prepare_data():
	train_data=read_data('Dataset/TrainImages.txt')
	test_data=read_data('Dataset/TestImages.txt')
	write_to_csv('Dataset/train.csv', train_data)
	write_to_csv('Dataset/test.csv', test_data)	

def createModel(train_img_num, batch_size):
    name="MIT-Xception-avg-depthw-constraints-512"

    #clear backend
    keras.backend.clear_session() 
    shape=(512,512,3)
    input_tensor=keras.Input(shape=shape)
    base_model=keras.applications.Xception(input_tensor=input_tensor,weights='imagenet',include_top=False)
    avg=keras.layers.AveragePooling2D(3,padding='valid')(base_model.output)
    depthw=keras.layers.DepthwiseConv2D(5,
                                          depthwise_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.01),
                                          bias_initializer=keras.initializers.Zeros(),depthwise_constraint=keras.constraints.NonNeg())(avg)
    flat=keras.layers.Flatten()(depthw)
    preds=keras.layers.Dense(67,activation='softmax',
                              kernel_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.01),
                              bias_initializer=keras.initializers.Zeros(),)(flat)
    model=keras.Model(inputs=base_model.input, outputs=preds)  

    for layer in model.layers:
      layer.trainable = True
    # filepath="models/cnn/%s-{epoch:02d}-{val_accuracy:.4f}.hdf5"%name
    # #creating checkpoint to save the best validation accuracy
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=False, mode='max') 
    # callbacks_list = [checkpoint]
    
    #Determine adaptive learning rate with an initialization value of 0.045 and decay of 0.94 every two epochs.
    lr_schedule =keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.045,
        decay_steps=2*int(train_img_num/batch_size),
        decay_rate=0.94,
        staircase=True)

    optimizer=keras.optimizers.SGD(momentum=0.9,learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def draw_metrics(path, name):
	hist=pd.read_csv('models/cnn/MIT-Xception-avg-depthw-constraints-512.csv', header=None)

	plt.plot(hist.loc[1, 1:])
	plt.plot(hist.loc[3, 1:])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(f'{path}{name}-accuracy.png', bbox_inches='tight')
	plt.clf()
	plt.close()

	plt.plot(hist.loc[0, 1:])
	plt.plot(hist.loc[2, 1:])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(f'{path}{name}-loss.png', bbox_inches='tight')
	plt.clf()
	plt.close()

def plot_confusion_matrix(cm, classes, path, name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {name}')
    # plt.show()
    plt.savefig(f'{path}{name}-cm.png', bbox_inches='tight')
    plt.clf()
    plt.close()

def create_confusion_matrix(name, model, validation_generator):
	# make predictions
	y_predict=model.predict(validation_generator) 
	y_pred=np.argmax(y_predict, axis=1)
	y_true=validation_generator.classes
	cm=confusion_matrix(y_true, y_pred)

	report = classification_report(y_true, y_pred, target_names=list(validation_generator.class_indices.keys()),output_dict=True)
	df_classification_report = pd.DataFrame(report).transpose()
	# df_classification_report = pd.DataFrame(report)
	accuracy_report = df_classification_report.tail(3)
	# accuracy_report = df_classification_report
	accuracy_report.to_csv('models/cnn/'+name+'_report.csv')

	df_classification_report.drop(df_classification_report.tail(3).index, inplace=True)
	df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
	df_classification_report.to_csv('models/cnn/'+name+'train_classification_report.csv')


	plot_confusion_matrix(cm, list(validation_generator.class_indices.keys()), 'models/cnn/', 'test-'+name)
	# plot_confusion_matrix(cm, list(range(len(validation_generator.class_indices.keys()))), 'models/cnn/', 'test-'+name)

def save_eval_metrics():
	self.accuracy = accuracy_score(y_true, y_pred)
	self.precision = precision_score(y_true, y_pred, average='weighted')
	self.recall = recall_score(y_true, y_pred, average='weighted')
	self.f1_score = f1_score(y_true, y_pred, average='weighted')

def run(train):
	name="MIT-Xception-avg-depthw-constraints-512"
	prepare_data()
	#Set data augmentation techniques
	train_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,vertical_flip=True
	                                                             ,zoom_range=0.2,rotation_range=360
	                                                             ,width_shift_range=0.1,height_shift_range=0.1
	                                                             ,channel_shift_range=50
	                                                             ,brightness_range=(0,1.2)
	                                                             ,preprocessing_function=keras.applications.imagenet_utils.preprocess_input)

	test_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.imagenet_utils.preprocess_input)

	train_df = pd.read_csv("Dataset/train.csv")
	test_df = pd.read_csv("Dataset/test.csv")

	#Create Data augmentation techniques
	batch_size=15
	train_generator = train_datagen.flow_from_dataframe(
	      dataframe=train_df,
	      directory='../data',
	      x_col="filename",
	      y_col="class",
	      target_size=(512, 512),
	      batch_size=batch_size,
	      class_mode='categorical',shuffle=True)

	validation_generator = test_datagen.flow_from_dataframe(
	        dataframe=test_df,
	        directory='../data',
	        x_col="filename",
	        y_col="class",
	        target_size=(512, 512),
	        batch_size=batch_size,
	        class_mode='categorical',shuffle=False)

	train_img_num=len(train_generator.filenames)
	model=createModel(train_img_num, batch_size)
	model.summary()

	if(train):

		# filepath="models/cnn/%s-{epoch:02d}-{val_accuracy:.4f}.hdf5"%name
		filepath="models/cnn/%s-{epoch:02d}.hdf5"%name
		#creating checkpoint to save the best validation accuracy
		checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=False, mode='max') 
		callbacks_list = [checkpoint]

		train_start_time = time.time()
		hist=model.fit_generator(train_generator, epochs=130,validation_data=validation_generator,shuffle=True,callbacks=callbacks_list) #start training
		# hist=model.fit(train_generator, epochs=1,validation_data=validation_generator,shuffle=True,callbacks=callbacks_list) #start training
		train_time = time.time() - train_start_time

		#write reports
		with open('models/cnn/{}.csv'.format(name), mode='w',newline='') as csv_file: 
		  csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		  for key in hist.history:
		    data=[key]
		    data.extend(hist.history[key])
		    csv_writer.writerow(data)
		print("Training finished. Reports saved!")
	else:
		# load weights of the best model
		model.load_weights("models/cnn/MIT-Xception-avg-depthw-constraints-512-64.hdf5")
	# print('Evaluating...')
	# draw_metrics('models/cnn/', name)
	# # evaluate model
	# loss, accuracy=model.evaluate(validation_generator)
	# print('Loss: '+str(loss)+'\nAccuracy: '+str(accuracy))

	print('Confusion Matrix Plotting...')
	create_confusion_matrix(name, model, validation_generator)

	print('Task Complete!!!')


if __name__ == '__main__':
	train=True

	if len(sys.argv) == 2:
		if sys.argv[1] == 'load':
			train=False

	run(train)