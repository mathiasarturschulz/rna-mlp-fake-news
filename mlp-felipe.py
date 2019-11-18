
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
​
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
​
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
​
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
​
# Any results you write to the current directory are saved as output.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
%matplotlib inline




train_dataset_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_dataset_df.head()




# get the correspondent numpy array of the data
train_dataset_np = train_dataset_df.values
​
train_labels = train_dataset_np[:,0]
train_pixels = train_dataset_np[:,1:] / 255.0
​
train_labels_one_hot = to_categorical(train_labels)
x_train, x_val, y_train, y_val = train_test_split(train_pixels, train_labels_one_hot, test_size=0.25, random_state=0)





y_train
print('Shape complete dataset: ', train_dataset_np.shape)
​
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
​
print('x_val shape: ', x_val.shape)
print('y_val shape: ', y_val.shape)





 
print('Images on the training set: ')
for i in range(0,6):
    plt.subplot(330 + i + 1)
    index = np.random.choice(x_train.shape[0])
    plt.imshow(x_train[index, : :].reshape(28,28), cmap='gray')


print(train_labels)


print(train_labels_one_hot)


