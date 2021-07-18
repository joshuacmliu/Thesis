import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle

datadir = "D:/Python/Tensorflow/BreaKHis_v1/BreaKHis_v1/400x"
categories = ["Benign", "Malignant"]

# for category in categories:
#     path = os.path.join(datadir, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path,img))
#         plt.imshow(img_array, cmap='gray')
#         plt.show()
#         break
#     break

training_images = []
training_labels = []
#training_data = []

#def create_training_data():
for category in categories:
    path = os.path.join(datadir, category)
    class_num = categories.index(category)
    for i, img in enumerate(os.listdir(path)):
        print(f'Loading {i}/{len(os.listdir(path))}')
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
            img_array_RGB = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img_array_RGB, (200, 200))
            training_images.append(new_array)
            training_labels.append(class_num)
        except Exception as e:
            print(e)

#create_training_data()
training_images = np.stack(training_images)
training_labels = np.stack(training_labels)

random_index = np.arange(0, len(training_images))
random.shuffle(random_index)
shuffled_images = training_images[random_index]
shuffled_labels = training_labels[random_index]

np.save("X400.npy", shuffled_images)
np.save("y400.npy", shuffled_labels)

#X = []
#y = []

# for features, label in training_data:
#     X.append(features)
#     y.append(label)

# X = np.array(X).reshape(-1, 700, 460, 3)

# pickle_out = open("X_BH.pickle" "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y_BH.pickle" "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()  

# pickle_in = open("X_BH.pickle" "rb")
# X=pickle.load(pickle_in)
