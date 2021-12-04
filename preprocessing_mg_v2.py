import csv
import pandas as pd
import time
import numpy as np
from glob import glob
from numpy import asarray
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

# get the image file names from the image-name-to-label-name mapping-file called train.csv
train_labels = np.loadtxt('data/train.csv', delimiter=',', skiprows=1, usecols=0, dtype=str)

# get the labels from the image-name-to-label-mapping file
# uncomment next line if not using complex labels
#train_labels_without_complex = pd.read_csv("data/train.csv", delimiter=",")


# put the image name files into a pandas dataframe
train_labels = pd.DataFrame(train_labels, columns = ['image_name'])


# complex label means unhealthy leaves with too many diseases to classify visually.
# complex can co-occur with other labels, but we are removing for our purposes.
# if not using complex labels, can uncomment next line
#train_labels = train_labels.loc[~train_labels_without_complex['labels'].str.contains('complex', regex=False)]

# turn image file name into numpy array
train_labels_column = train_labels.iloc[:,0]
train_labels = train_labels_column.values


start = time.time()
count = 0
dim = 100
sample_size = 18632


# get data in the order of the labels
for i_list in batch(train_labels, chunk):

    im_arr = []
    #temp_label = []
    for i in i_list:
        count += 1
        image = Image.open(img_dir + i)
        # this should be resizing the files to all the same dimensions
        image.thumbnail((dim, 2 * dim))
        im = np.array(image)

        # limiting to only images that fit shape for now - im.shape[0] seems to be the only that varies
        #  33 for 50, 67 for 100, 134 for 200
        if im.shape[0] == 67:
            # Keep these label lines.
            # Use flatten to append as new row
            im2 = im.flatten()
            print(count, i)
            # we don't need this we can flatten.
            #im = list(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]).tolist())

            # append list as new row to csv
            # write the rgb array to train_data.csv file
            with open("train_data.csv", "a+") as g:
                write = csv.writer(g)
                write.writerow(im2)
            # create file which will have list of labels which correspond to the
            # files we  have kept.
            print(i, file=open("label_keep.txt", "a+"))

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


# write a new file full of labels we will keep
def writelabels():

    # load list of labels I want to keep as a pandas dataframe
    label_list = pd.read_csv('label_keep.txt')

    label_list = label_list.iloc[:, 0]. tolist()

    # now load the full image-name-to-label mapping file
    full_labels = pd.read_csv('data/train.csv')
    # create a pandas dataframe for full image-name-to-label mapping file
    data = pd.DataFrame(full_labels)
    # keep only the rows from the full dataframe where the first column equals the
    # name of the files we want to keep.
    data = data[data['image'].isin(label_list)]

    # save a csv file which contains only the corresponding labels for the image files
    # we decided to keep
    data.to_csv('data/train_labels_final.csv', index=False)
    print(len(label_list))

    # handle space delimited labels by splitting label file when there are multiple labels

    with open('data/train_labels_final.csv') as infile, open('data/train_labels_final_single.csv', 'w') as outfile_two:
        for line in infile:
            outfile_two.write(" ".join(line.split()).replace(' ', ','))
            outfile_two.write("\n")


# call the write labels function which prints the names and labels for all the image
# files we want to keep
writelabels()

end = time.time()
print('Process Complete')
total_time = round(end - start, 2)
print("Total time was: " + str(total_time) + " seconds.")
