import csv
import pandas as pd
import time
import numpy as np
from glob import glob
from numpy import asarray
from PIL import Image
import os
from natsort import os_sorted

start = time.time()

# batch loader chunk size
chunk = 50

""" Image Preprocessing Parameters

raw images are first center-cropped, and then resized to the target (processed) image size.
the centered crop box has the same aspect ratio as the target.
for each raw image, the crop box is initially scaled up to maximize coverage of the raw image, limited by
image height and/or width.
if crop_box_scale < 100, the crop box is then scaled down to <crop_box_scale>% of its initial value.
e.g., if crop_box_scale=50, the crop box height and width are halved and cover only 25% of the pixels of
the original crop box (remaining centered on the raw image).
the cropped image is then resized to the target image size.
"""

# target image size
# target_width = 100          # mili's original 100x67
# target_height = 67

target_width = 100
target_height = 67
crop_box_scale = 71         # use 25% of the maximum crop box area
resample_filter = Image.BILINEAR     # resize to target size with the PIl.Image.BILINEAR filter
use_label_directories = True
image_limit = 0       # number images to process (if 0, process all iamges)

""" Calculate unique key for these parameter selections. """

if resample_filter is None:
    resample_filter_key = None
    resample_filter_name = "NO"
elif resample_filter == Image.BILINEAR:
    resample_filter_key = 'BL'
    resample_filter_name = "bilinear"
else:
    raise ValueError(f"unrecognized resample_filter_key: '{resample_filter}'")
image_limit_key = f"_{image_limit}" if image_limit > 0 else ""
resize_key = f"{crop_box_scale}_{target_width}_{target_height}_{resample_filter_key}{image_limit_key}"

""" Get directory and file paths (names). """

train_source_dir = "data"
train_dest_dir = f"data/{resize_key}"
if not os.path.exists(train_dest_dir):
    os.makedirs(train_dest_dir)
    print("Created new output directory:", train_dest_dir)
else:
    print("Reusing existing output directory:", train_dest_dir)

# create batching function
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# get the image file names from the image-name-to-label-name mapping-file called train.csv
train_images = np.loadtxt(f"{train_source_dir}/train.csv", delimiter=',', skiprows=1, usecols=0, dtype=str)

# put the image name files into a pandas dataframe
train_images = pd.DataFrame(train_images, columns = ['image_name'])

# turn image file name into numpy array
train_images_column = train_images.iloc[:,0]
train_images = train_images_column.values

# get the image labels
train_labels = np.loadtxt(f"{train_source_dir}/train.csv", delimiter=',', skiprows=1, usecols=1, dtype=str)
train_labels = pd.DataFrame(train_labels, columns = ['label'])
train_labels_column = train_labels.iloc[:,0]
train_labels = train_labels_column.values

train_data_csv_filename = f"{train_dest_dir}/train_data.csv"
train_data_csv_fp = open(train_data_csv_filename, "w")
train_data_csv_writer = csv.writer(train_data_csv_fp)

train_images_csv_filename = f"{train_dest_dir}/train_images.csv"
train_images_csv_fp = open(train_images_csv_filename, "w")

image_cnt = image_limit if image_limit > 0 else train_images.shape[0]

print(f"Converting {image_cnt} raw images from {train_source_dir}/train_images/:")
print(f" center cropped to {crop_box_scale}% of the largest possible bounding box matching the target's aspect ratio,")
print(f" resized to {target_width}x{target_height} using {resample_filter_name} resampling,")
if image_limit > 0:
    print(" limited to first {image_limit} images,")
if use_label_directories:
    print(f" grouped by labels into subdirectories,")
print(f" and written to {train_dest_dir}/train_images/:")

count = 0
# get data in the order of the labels
for i_list, l_list in zip(batch(train_images, chunk), batch(train_labels, chunk)):
    for image_filename, labels in zip(i_list, l_list):
        count += 1
        raw_image_filename = f"{train_source_dir}/train_images/{image_filename}"
        image = Image.open(raw_image_filename)

        # calculate crop box for target size within this image
        crop_width_scale = image.width // target_width
        crop_height_scale = image.height // target_height
        crop_scale = min(crop_width_scale, crop_height_scale)
        crop_width = target_width * crop_scale * crop_box_scale // 100
        crop_height = target_height * crop_scale * crop_box_scale // 100
        half_crop_width = crop_width // 2
        half_crop_height = crop_height // 2
        half_width = image.width // 2
        half_height = image.height // 2
        crop_box = (half_width-half_crop_width, half_height-half_crop_height, half_width+half_crop_width, half_height+half_crop_height)
        image_resized = image.resize((target_width, target_height), box=crop_box, resample=resample_filter)

        sorted_train_dest_dir = f"{train_dest_dir}/train_images"
        if use_label_directories:
            labels = labels.replace(' ', '__') + '__'
            sorted_train_dest_dir = f"{sorted_train_dest_dir}/{labels}"
            if not os.path.exists(sorted_train_dest_dir):
                os.makedirs(sorted_train_dest_dir)
                #print("Created new output directory:", sorted_train_dest_dir)

        target_image_filename = f"{sorted_train_dest_dir}/{image_filename}"
        image_resized.save(target_image_filename)

        im = np.array(image_resized)
        im2 = im.flatten()
        train_data_csv_writer.writerow(im2)
        print(image_filename, file=train_images_csv_fp)
        image.close()
        image_resized.close()

        if count % 100 == 0:
            print(f"\rConverted {count} images...", end='')
        if count == image_limit:
            break
    if count == image_limit:
        break

print(f"\rConverted {count} images.")

train_data_csv_fp.close()
train_images_csv_fp.close()

# write a new file full of labels we will keep
def writelabels():

    # load list of labels I want to keep as a pandas dataframe
    label_list = pd.read_csv(train_images_csv_filename, header=None)

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
    data.to_csv(f"{train_dest_dir}/train_labels_final.csv", index=False)
    print(len(label_list))

    # handle space delimited labels by splitting label file when there are multiple labels

    with open(f"{train_dest_dir}/train_labels_final.csv") as infile, open(f"{train_dest_dir}/train_labels_final_single.csv", 'w') as outfile_two:
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
pass
