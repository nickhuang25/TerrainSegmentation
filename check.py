from PIL import Image
import numpy as np
import os


dataset_path = "camera/mask/"

counter = 0
zero_counter = 0
one_counter = 0
two_counter = 0
for filename in os.listdir(dataset_path):    
    filename = dataset_path + filename
    img = Image.open(filename)
    np_arr = np.asarray(img)
    for i in range(np_arr.shape[0]):
        for j in range(np_arr.shape[1]):
            if np_arr[i][j] == 0:
                zero_counter += 1
            elif np_arr[i][j] == 1:
                one_counter += 1
            elif np_arr[i][j] == 2:
                two_counter += 1

    total = zero_counter + one_counter + two_counter
    print("Image " + str(counter))
    print(zero_counter, one_counter, two_counter, total)
    zero_counter = 0
    one_counter = 0
    two_counter = 0
    
    counter += 1