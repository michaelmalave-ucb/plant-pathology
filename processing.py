import csv
import time
import numpy as np
from glob import glob
from PIL import Image


# create batching function
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# initialize configurations for bathing and ideal size
chunk = 50
img_dir = 'data/train_images/'
image_files = glob(img_dir + '*.jpg')
train_labels = np.loadtxt('data/train.csv', delimiter=',', skiprows=1, usecols=0, dtype=str)
start = time.time()
count = 0
square = 5
# get data in the order of the labels
for i_list in batch(train_labels, chunk):

    im_arr = []
    for i in i_list:
        count += 1
        image = Image.open(img_dir + i)
        # this should be resizing the files to all the same dimensions
        image.thumbnail(size=(200, 400), resample=1)
        im = np.array(image)
        # limiting to only images that fit shape for now
        if im.shape[0] == 134:
            im = list(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]).tolist())
            # append list as new row to csv
            with open("train_data.csv", "a+") as f:
                write = csv.writer(f)
                write.writerow(im)
        else:
            pass
        # show some sample items
        if count < 5:
            image.show()
        # track progress to completion
        if count % 50 == 0:
            print("Completed items: " + str(count))
        image.close()
    # This limits to a subset of the data to be added to the file
    if count == 500:
        break


end = time.time()
print('Process Complete')
total_time = round(end - start, 2)
print("Total time was: " + str(total_time) + " seconds.")