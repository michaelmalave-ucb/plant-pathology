import csv
import time
import numpy as np
from glob import glob
# adding line for import of asarray and pandas if needed
from numpy import asarray
import pandas as pd
from PIL import Image




# handle multiple, space delimited labels
#with open('data/train.csv') as infile, open('data/train_labels.csv', 'w') as outfile:
    #for line in infile:
        #outfile.write(" ".join(line.split()).replace(' ', ','))
        #outfile.write("\n")



# create batching function
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# initialize configurations for bathing and ideal size
chunk = 50
img_dir = 'data/train_images/'
image_files = glob(img_dir + '*.jpg')

# get the image file names from the image name to label name mapping file called train.csv
train_labels = np.loadtxt('data/train.csv', delimiter=',', skiprows=1, usecols=0, dtype=str)


# get the labels from the image name to label mapping file
train_labels_without_complex = pd.read_csv("data/train.csv", delimiter=",")

# put the image name files into a pandas dataframe
train_labels = pd.DataFrame(train_labels, columns = ['image_name'])

# get all the image file names which do not map back to a label which has complex label.
# complex label means unhealthy leaves with too many diseases to classify visually.
# complex can co-occur with other labels, but we are removing for our purposes.
train_labels = train_labels.loc[~train_labels_without_complex['labels'].str.contains('complex', regex=False)]

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
    temp_label = []
    for i in i_list:
        count += 1
        image = Image.open(img_dir + i)
        # this should be resizing the files to all the same dimensions
        image.thumbnail((dim, 2 * dim))
        im = np.array(image)
        #print(im.shape[0])
        # limiting to only images that fit shape for now - im.shape[0] seems to be the only that varies
        #  33 for 50, 67 for 100, 134 for 200
        if im.shape[0] == 67:
            # Keep these label lines
            # Use flatten to append as new row
            im2 = im.flatten()
            #print(i, file=open("output.txt", "w+"))
            print(count, i)
            # we don't need this we can flatten.
            #im = list(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]).tolist())
            # append list as new row to csv
            #with open("train_data_test.csv", "a+") as f:
                #write = csv.writer(f)
                #write.writerow(im)
            # write the rgb array to train_data.csv file
            with open("train_data.csv", "a+") as g:
                write = csv.writer(g)
                write.writerow(im2)
            # create file which will have list of labels which correspond to the
            # files we kept.
            print(i, file=open("label_keep.txt", "a+"))



        else:
            pass
        # show some sample items
        """if count < 5:
            image.show()"""
        # track progress to completion
        if count % 10 == 0:
            print("\lCompleted items: " + str(count), end='')
        image.close()
    # This limits to a subset of the data to be added to the file
    if count == sample_size:
        break


# Write a new file full of labels we will keep
def writelabels():

    #load list of labels I want to keep
    label_list = pd.read_csv('label_keep.txt')

    label_list = label_list.iloc[:, 0]. tolist()


    full_labels = pd.read_csv('data/train.csv')
    data = pd.DataFrame(full_labels)
    data = data[data['image'].isin(label_list)]
    data.to_csv('data/train_labels_final.csv', index=False)
    print(len(label_list))

writelabels()
end = time.time()
print('Process Complete')
total_time = round(end - start, 2)
print("Total time was: " + str(total_time) + " seconds.")
