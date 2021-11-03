import csv
import time
import numpy as np
from glob import glob
from PIL import Image


# handle multiple, space delimited labels
with open('data/train.csv') as infile, open('data/train_labels.csv', 'w') as outfile:
    for line in infile:
        outfile.write(" ".join(line.split()).replace(' ', ','))
        outfile.write("\n")


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
dim = 100
sample_size = 2000

# get data in the order of the labels
for i_list in batch(train_labels, chunk):

    im_arr = []
    for i in i_list:
        count += 1
        image = Image.open(img_dir + i)
        # this should be resizing the files to all the same dimensions
        image.thumbnail((dim, 2 * dim))
        im = np.array(image)
        # print(im.shape[0])
        # limiting to only images that fit shape for now - im.shape[0] seems to be the only that varies
        #  33 for 50, 67 for 100, 134 for 200
        if im.shape[0] == 67:
            im = list(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]).tolist())
            # append list as new row to csv
            with open("train_data.csv", "a+") as f:
                write = csv.writer(f)
                write.writerow(im)
        else:
            pass
        # show some sample items
        """if count < 5:
            image.show()"""
        # track progress to completion
        if count % 500 == 0:
            print("Completed items: " + str(count))
        image.close()
    # This limits to a subset of the data to be added to the file
    if count == sample_size:
        break


end = time.time()
print('Process Complete')
total_time = round(end - start, 2)
print("Total time was: " + str(total_time) + " seconds.")


