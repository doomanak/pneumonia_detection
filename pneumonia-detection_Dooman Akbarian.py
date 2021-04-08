#%% 
###https://github.com/krishnaik06/Predicitng-Lungs-Disease-/blob/master/Transfer%20Learning%20VGG%2016.ipynb
# import the libraries as shown below
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import os
#import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from mlxtend.plotting import plot_confusion_matrix
import pickle
import re
# Deep Learning - Keras - Pretrained Models
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

# Deep Learning - Keras - Layers
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
#from keras.layers.pooling import _GlobalPooling1D

# Deep Learning - Keras - Model Parameters and Evaluation Metrics
from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy

# Deep Learning - Keras - Visualisation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

import time

#get_ipython().run_line_magic('matplotlib', 'qt')
#%% ############################################################################
 # re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = '../chest_xray/train'
valid_path = '../chest_xray/val'
test_path = '../chest_xray/test'     
classes = ['NORMAL','PNEUMONIA']#os.listdir(train_path) 

batch_size1=32
num_eps=50
    
#%% ##########################################Tran/Test generation ##########################################
# Use the Image Data Generator to import the images from the dataset

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = batch_size1,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = batch_size1,
                                            class_mode = 'categorical',
                                            shuffle=False)

val_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = batch_size1,
                                            class_mode = 'categorical')
#%% ##########################################Model Definition##########################
def get_model(model_name, input_tensor=Input(shape=(96,96,3)), num_class=2): ##Modified by Dooman
    inputs = Input(input_shape)
    
    if model_name == "Xception":
        base_model = Xception(include_top=False, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(include_top=False, input_shape=input_shape)
    elif model_name == "ResNet101":
        base_model = ResNet101(include_top=False, input_shape=input_shape)
    elif model_name == "InceptionV3":
        base_model = InceptionV3(include_top=False, input_shape=input_shape)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    elif model_name == "DenseNet201":
        base_model = DenseNet201(include_top=False, input_shape=input_shape)
    elif model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, input_tensor=input_tensor) ##Modified by Dooman
    elif model_name == "NASNetLarge":
        base_model = NASNetLarge(include_top=False, input_tensor=input_tensor)
    if model_name == "VGG16":    
        base_model = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        
            
    for layer in base_model.layers:
        layer.trainable = False
           
    x = base_model(inputs)
    
    output1 = GlobalMaxPooling2D()(x)
    output2 = GlobalAveragePooling2D()(x)
    output3 = Flatten()(x)
    
    outputs = Concatenate(axis=-1)([output1, output2, output3])
    
    outputs = Dropout(0.5)(outputs)
    outputs = BatchNormalization()(outputs)
    
    if num_class>1:
        outputs = Dense(num_class, activation="softmax")(outputs)
    else:
        outputs = Dense(1, activation="sigmoid")(outputs)
        
    model = Model(inputs, outputs)
    
    model.summary()
    
    
    return model

input_shape = (224, 224, 3)

num_class = len(classes)#2

input_tensor=Input(shape=(224, 224, 3))

#%% #######################################
#Plot step    
fnt_size=32
lnewidth=4
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}
####
model_lists=['NASNetLarge','InceptionResNetV2','InceptionV3','Xception','ResNet101']#['ResNet50','InceptionV3','Xception']#['ResNet50','VGG16','NASNetMobile','NASNetLarge','InceptionV3','DenseNet201','InceptionResNetV2','Xception']
####
fig_loss_test=plt.figure(figsize=(5, 4), dpi=300)
ax_loss_test=fig_loss_test.gca()
##
fig_loss_train=plt.figure(figsize=(5, 4), dpi=300)
ax_loss_train=fig_loss_train.gca()
##
fig_acc_test=plt.figure(figsize=(5, 4), dpi=300)
ax_acc_test=fig_acc_test.gca()
##
fig_acc_train=plt.figure(figsize=(5, 4), dpi=300)
ax_acc_train=fig_acc_train.gca()
##
fig_roc=plt.figure(figsize=(6, 6), dpi=300)
ax_roc=fig_roc.gca()
##
color_codes=['b','y','r','orange','g']
####
i=0
for model_i in model_lists:
    tic = time.process_time()
    Model_performance = open("%s_Performance.txt" % model_i, "+w")
    model = get_model(model_name=model_i, input_tensor=input_tensor, num_class=num_class)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    r = model.fit_generator(training_set,validation_data=test_set,epochs=num_eps,steps_per_epoch=len(training_set),validation_steps=len(test_set),use_multiprocessing=False)
    model.save("%s_Model.h5" % model_i)
    ######## WModel Evaluation ########
    y_pred = model.predict_generator(test_set, steps=len(test_set), verbose=1) 
    y_pred_prob=y_pred[:,1] 
    y_pred = y_pred.argmax(axis=-1)#0:Normal, 1:Pnemonia
    y_true=test_set.classes
    precision = precision_score(y_true, y_pred) 
    recall = recall_score(y_true, y_pred) 
    f1 = f1_score(y_true, y_pred)
    cls_report_print = classification_report(y_true, y_pred, target_names=classes)
    CM = confusion_matrix(y_true, y_pred)
    false_positive_rate, recalls, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(false_positive_rate, recalls)
    ######## Write the outputs ########            
    Model_performance.write("-"*60) 
    Model_performance.write("\n")
    Model_performance.write("Report of %s Model Performance\n"% model_i)
    Model_performance.write("-"*60)
    Model_performance.write("\n") 
    Model_performance.write("Accuracy    :%s\n" % str(format(np.mean(y_true==y_pred)*100, '.2f')))
    Model_performance.write("Precision   :%s\n" % str(format(precision*100, '.2f')))
    Model_performance.write("Recall      :%s\n" % str(format(recall*100, '.2f')))
    Model_performance.write("F1-Score    :%s\n" % str(format(f1*100, '.2f')))
    Model_performance.write("AUC         :%s\n" % str(format(roc_auc, '.2f')))
    Model_performance.write("-"*60)
    Model_performance.write("\n")
    Model_performance.write(cls_report_print)
    Model_performance.write("-"*60)
    Model_performance.close()
    ######## Save the Figs ########
    # plot the loss
    ax_loss_train.plot(list(range(1, num_eps+1)),r.history['loss'], color=color_codes[i],label='%s'% model_i)
    ax_loss_test.plot(list(range(1, num_eps+1)),r.history['val_loss'], color=color_codes[i],label='%s'% model_i)
    # plot the accuracy
    ax_acc_train.plot(list(range(1, num_eps+1)),r.history['acc'], color=color_codes[i],label='%s'% model_i)
    ax_acc_test.plot(list(range(1, num_eps+1)),r.history['val_acc'], color=color_codes[i],label='%s'% model_i)
    # plot the Confusion Matrix   
    plt.rcParams.update({'font.size': 20})
    fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(15,15), cmap=plt.cm.Blues)
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.ylim([-0.5,1.5])
    plt.xlabel('Predicted lable', fontname="Arial",fontsize= 20)
    plt.ylabel('True lable' , fontname="Arial",fontsize= 20)
    fig.savefig('%s_Confusion_Matrix'% model_i)
    plt.rcParams.update({'font.size': 10})
#     # plot the ROC
    ax_roc.plot(false_positive_rate, recalls,  color=color_codes[i], label = '%s AUC = %0.3f' % (model_i,roc_auc))
####
    i+=1
    toc = time.process_time()
    time_elapsed=toc-tic
    print('**********Elapsed time:*********',time_elapsed)
#####    
ax_loss_test.set_xlabel('Epoch', fontname="Arial",fontsize= 14)
ax_loss_test.set_ylabel('Loss' , fontname="Arial",fontsize= 14)
ax_loss_test.set_xticks(np.arange(0, num_eps+1, num_eps/5))
ax_loss_test.set_xlim([1.0,num_eps])
ax_loss_test.legend() 
fig_loss_test.savefig('Loss_test')
#####
ax_loss_train.set_xlabel('Epoch', fontname="Arial",fontsize= 14)
ax_loss_train.set_ylabel('Loss' , fontname="Arial",fontsize= 14)
ax_loss_train.set_xticks(np.arange(0, num_eps+1, num_eps/5))
ax_loss_train.set_xlim([1.0,num_eps])
ax_loss_train.legend() 
fig_loss_train.savefig('Loss_train')
#####
ax_acc_test.set_xlabel('Epoch', fontname="Arial",fontsize= 14)
ax_acc_test.set_ylabel('Accuracy' , fontname="Arial",fontsize= 14)
ax_acc_test.set_xticks(np.arange(0, num_eps+1, num_eps/5))
ax_acc_test.set_xlim([1.0,num_eps])
ax_acc_test.set_yticks(np.arange(0, 2, 0.2))
ax_acc_test.set_ylim([0,1])
ax_acc_test.legend() 
fig_acc_test.savefig('Accuracy_test')
#####
ax_acc_train.set_xlabel('Epoch', fontname="Arial",fontsize= 14)
ax_acc_train.set_ylabel('Accuracy' , fontname="Arial",fontsize= 14)
ax_acc_train.set_xticks(np.arange(0, num_eps+1, num_eps/5))
ax_acc_train.set_xlim([1.0,num_eps])
ax_acc_train.set_yticks(np.arange(0, 2, 0.2))
ax_acc_train.set_ylim([0,1])
ax_acc_train.legend() 
fig_acc_train.savefig('Accuracy_train')
#####
ax_roc.legend(loc='lower right')
ax_roc.plot([0,1], [0,1], 'k--')
ax_roc.set_xlim([0.0,1.0])
ax_roc.set_ylim([0.0,1.0])
ax_roc.set_ylabel('True Positive Rate',fontsize= 14)
ax_roc.set_xlabel('False Positive Rate',fontsize= 14)
fig_roc.savefig('ROC')













