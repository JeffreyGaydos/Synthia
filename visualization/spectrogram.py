#This file is assumed to be run from the root of this repo, i.e. "py .\visualization\spectrogram.py"

print("importing...")
import os
import pathlib

#NOTE: These bois are beefy, give it a couple seconds to load everything up
print("importing matplotlin.pyplot...")
import matplotlib.pyplot as plt
print("importing numpy...")
import numpy as np
print("importing seaborn...")
import seaborn as sb
print("importing tensorflow...")
import tensorflow as tf

# from tensorflow.keras import layers
# from tensorflow.keras import models

# from IPython import display
print("imports complete")

dataset = pathlib.Path("dataset")

tfdataset_train, tfdataset_validate = tf.keras.utils.audio_dataset_from_directory(
    directory=dataset,
    batch_size=3,
    validation_split=0.1, # Usually 20 for 20% of data is saved for validation
    seed=0,
    output_sequence_length=150000,
    subset='both'
)

classes = np.array(tfdataset_train.class_names)

print("Classifications: ", classes)

print(tfdataset_train.element_spec) #looks like our .wavs are the exact same spec as the tutorial

# def removeChannelAxis(audio, labels):
#     audio = tf.squeeze(audio, axis=-1)
#     return audio, labels

# tfdataset_train = tfdataset_train.map(removeChannelAxis, tf.data.AUTOTUNE)
# tfdataset_validate = tfdataset_validate.map(removeChannelAxis, tf.data.AUTOTUNE)
somethingsAudios = []
somethingsLabels = []
#skipping some steps...
for somethingAudio, somethingLabel in tfdataset_train:
    somethingsAudios.append(somethingAudio)
    somethingsLabels.append(somethingLabel)

for somethingAudio, somethingLabel in tfdataset_validate:
    somethingsAudios.append(somethingAudio)
    somethingsLabels.append(somethingLabel)

def plot_comparison_all():
    plt.figure(1, [20, 16], None, 'w', 'darkcyan', True)
    #plot all
    rows = 7
    cols = 5
    bound = 0.25
    j = 0
    # for i in range(rows * cols):
    for i in range(len(somethingsAudios)):
        plt.subplot(rows, cols, j + 1)
        j = j + 1
        plt.plot(somethingsAudios[i][0])
        plt.title(classes[somethingsLabels[i][0]])
        plt.yticks(np.arange(-1 * bound, 1 * bound, bound / 2))
        plt.ylim([-1 * bound, bound])
        plt.subplot(rows, cols, j + 1)
        j = j + 1
        plt.plot(somethingsAudios[i][1])
        plt.title(classes[somethingsLabels[i][1]])
        plt.yticks(np.arange(-1 * bound, 1 * bound, bound / 2))
        plt.ylim([-1 * bound, bound])
        plt.subplot(rows, cols, j + 1)
        j = j + 1
        plt.plot(somethingsAudios[i][2])
        plt.title(classes[somethingsLabels[i][2]])
        plt.yticks(np.arange(-1 * bound, 1 * bound, bound / 2))
        plt.ylim([-1 * bound, bound])

# plot_comparison_all()

def plot_comparison_by_class(classification, fignum):
    plt.figure(fignum, [20, 16], None, 'w', 'darkcyan', True)
    #plot specific class
    rows = 3
    cols = 1
    bound = 0.25
    j = 0
    # for i in range(rows * cols):
    for i in range(len(somethingsAudios)):
        if classes[somethingsLabels[i][0]] == classification:
            plt.subplot(rows, cols, j + 1)
            j = j + 1
            plt.plot(somethingsAudios[i][0])
            plt.title(classes[somethingsLabels[i][0]])
            plt.yticks(np.arange(-1 * bound, 1 * bound, bound / 2))
            plt.ylim([-1 * bound, bound])
        if classes[somethingsLabels[i][1]] == classification:
            plt.subplot(rows, cols, j + 1)
            j = j + 1
            plt.plot(somethingsAudios[i][1])
            plt.title(classes[somethingsLabels[i][1]])
            plt.yticks(np.arange(-1 * bound, 1 * bound, bound / 2))
            plt.ylim([-1 * bound, bound])
        if classes[somethingsLabels[i][2]] == classification:
            plt.subplot(rows, cols, j + 1)
            j = j + 1
            plt.plot(somethingsAudios[i][2])
            plt.title(classes[somethingsLabels[i][2]])
            plt.yticks(np.arange(-1 * bound, 1 * bound, bound / 2))
            plt.ylim([-1 * bound, bound])

for i in range(len(classes)):
    plot_comparison_by_class(classes[i], i + 2) # start at 2

print("wait up!")