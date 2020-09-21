import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from helper_functions import *
from sklearn.model_selection import train_test_split
from models import get_model


def main():
    setup_gpu()
    image_files,mask_files = load_data()
    X_train,X_test,y_train,y_test = train_test_split(image_files,mask_files,test_size=0.2)

    gen = Generator(X_train,5)
    model = get_model((128,128,3),2)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    gen = Generator(X_train,5)
    model.fit(gen,steps_per_epoch=48,epochs=15)

    model.save('wbc_segmentation_model.h5')

main()
