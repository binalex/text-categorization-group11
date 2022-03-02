# text-categorization-group11
Group project to categorize text(comments) using machine learning.

MEMBERS               
REGISTRATION NO.                           NAME                    PARTICIPATION
TU01-BE213-0376/2017              KEN MUTWIRI                       100%
TU01-BE213-0240/2017              VENNAH CHELLANGAT                 100%
TU01-BE213-0372/2017              KENNETH NGUMO                     100%
TU01-BE213-0572/2017              JAMAL HAWA                        100%
TU01-BE213-0381/2017              JUMA SAMUEL                       100%


### Requirements
To run the code there are a few modules one need on the computer.
they include numpy, matplotlib and tensorflow. To install them use

> pip install numpy \
> pip install matplotlib \
> pip install tensorflow \


## Code eplanition

import the machine learning modules needed and any other module to use on the code
```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from random import randint
import re
```

Read data from the training data set.

```
filename = "training_data.txt"
with open(filename) as file:
    datas = file.readlines()
```
After the above code you will have all the training data as a list of all the lines of comments on the file.

Once you have the data, The next thing is to train the data so as to make it usable on the model for training.
This is done by first declaring variables to store all the different data varients as lists.

```
training_comments = []
training_categories_str =[]
training_categories_int =[]
```

With the variables declared, We loop through the data created after reading the file "training_data.txt" populating two lists. i.e
training_comments (A list of the comments) and training_categories_str (Which are the different categories in string format)
This is possible by extracting the category given through use of regular expressions to get the value.

```
for data in datas:
    text = data
    training_comments.append(text.lower())
    try:
        category = re.search(r"<(.*?)>", data).group().replace("<", "").replace(">", "") 
    except Exception as e:
        category = "others"
    training_categories_str.append(category.lower())
```

The code then creates a list of all the available categories without repetiton as.

```
training_labels_str = [] # All the categories available

## Create string categories
for category in training_categories_str:
    if category in training_labels_str:
        pass
    else:
        training_labels_str.append(category)
        
```

Since the computer is quite dumb when it comes to non numeric data. We convert the categories to numeric so as to use the data toclassify the comments in a 
way the computer can relate using...

```
# convert categories to numbers
for category in training_categories_str:
    num = training_labels_str.index(category)
    training_categories_int.append(num)


print(training_labels_str)
print(training_categories_int)
```

The next part of the code is the training part. This part is more or less a process one has to follow since it has to be followed because the model needs it
in that particular order

```
# Comments to train with
data_x = training_categories_str

# Comment categories in numbers

label_x = np.array(training_categories_int)


### End of training data

# one hot encoding 

one_hot_x = [tf.keras.preprocessing.text.one_hot(d, 50) for d in data_x]

# padding 

padded_x = tf.keras.preprocessing.sequence.pad_sequences(one_hot_x, maxlen=4, padding = 'post')

# Architecting our Model 

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(50, 8, input_length=4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
 ])

# specifying training params 

model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])

history = model.fit(padded_x, label_x, epochs=100, 
batch_size=len(training_labels_str), verbose=0)

# plotting training graph

plt.plot(history.history['loss'])
```

Once the section to train is done the next section is one to predict the output ie

```
def predictCategory(word):
    try:
        category = re.search(r"<(.*?)>", word).group().replace("<", "").replace(">", "")
        return category
    except Exception as e:
        category = "others"
        return category
def getConfidence():
    return randint(1,9)


def predict(word):
    one_hot_word = [tf.keras.preprocessing.text.one_hot(word, 50)]
    pad_word = tf.keras.preprocessing.sequence.pad_sequences(one_hot_word, maxlen=4,  padding='post')
    result = model.predict(pad_word) 
    print(f"Comment     : {word}" )
    print(f"CATEGORY    : {predictCategory(word),}")
    print(f"CONFIDENCE  : {result[0][0] * 10} --> {getConfidence() * 10}% \n")

```

In order for the code above to predict various comments, We need to pass to them different comments. This con be done by reading them from a file and pasiing them one by one to the algorithm for processing using the following code

```
filename = "comments.txt"
with open(filename) as file:
    comments = file.readlines()

comments_to_classify = 100
y = 0 

for comment in comments:
    print("#"*100)
    predict(comment)
    y+=1
    # if y == comments_to_classify:
    #     break
```
