from PIL import Image
import numpy as np
import os


dataset_path = "camera/trav/"
output_dir = "camera/mask/"

for filename in os.listdir(dataset_path):
    filepath = dataset_path + filename
    img = Image.open(filepath).convert('L')
    np_arr = np.asarray(img)
    # for i in range(np_arr.shape[0]):
    #     for j in range(np_arr.shape[1]):
    #         if np_arr[i][j] == 0:
    #             print("found")
    
    arr_cpy = np_arr.copy()
    arr_cpy[np_arr == 29] = 1
    arr_cpy[np_arr == 255] = 2
    
    img = Image.fromarray(arr_cpy, 'L')
    filename = filename.replace("trav", "mask")
    img.save(output_dir+filename)
