

import pathlib      as path
import pprint       as pp
import json         as json
import random       as random
import os           as os
import operator     as op
import itertools    as itertools

import lmdb         as lmdb
import cv2          as cv2
import caffe        as caffe
import numpy        as np

from caffe.proto                import caffe_pb2
from collections                import defaultdict, Counter
from imblearn.over_sampling     import SMOTE
from imblearn.under_sampling            import RandomUnderSampler
from copy                       import deepcopy

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#                                       GLOBALS
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
REGENERATE_LMDB_DATABASES = True
# REGENERATE_LMDB_DATABASES = False

DATASET_PATH = path.Path("../segments-dataset/")
SEGMENT_IMAGE_WIDTH     = 128 
SEGMENT_IMAGE_HEIGHT    = 128 

def split_data( segments_descriptor
              , holdout_percent     = 0.3
              , rng_seed            = 0):
    random.seed(rng_seed)
    # training_sample_count = int(len(segments_descriptor) * (1 - holdout_percent))
    validation_sample_count = 10_000
    all_data_keys = list(segments_descriptor.keys())
    random.shuffle(all_data_keys)

    validation_data = [k for k in all_data_keys 
            if "is_flipped" not in segments_descriptor[k]][:validation_sample_count]
    training_data = set(all_data_keys) - set(validation_data)

    training_data = list(training_data)
    random.shuffle(training_data)

    validation_samples = {key: segments_descriptor[key] for key in validation_data}
    training_samples = {key: segments_descriptor[key] for key in training_data}
    
    return training_samples, validation_samples

def build_image_path(segment_descriptor):
    return DATASET_PATH / segment_descriptor["Segment Relative Path"]

def build_mask_path(segment_descriptor):
    return DATASET_PATH / segment_descriptor["Binary Segment Relative Path"]

def apply_image_transformations(image, image_mask, image_width, image_height):
    # We probably want to apply the mask to the input images somewhere in here. 
    image = cv2.bitwise_and(image, image, mask=image_mask)
    # image = equalize_image_histogram(image, image_width, image_height)
    image = cv2.resize(image, (image_width, image_height),
            interpolation=cv2.INTER_CUBIC)
    return image

def equalize_image_histogram(image, image_width, image_height):
    for channel_idx in range(3):
        image[:, :, channel_idx] = cv2.equalizeHist(image[:, :, channel_idx])
    return image

def build_lmdb(lmdb_path, segments_descriptor):
    image_output_directory = path.Path("./masked-images/")
    lmdb_db = lmdb.open(str(lmdb_path), map_size=int(1e12))
    with lmdb_db.begin(write=True) as lmdb_transaction:
        for sample_idx, sample in list(enumerate(segments_descriptor.values())):
            image_path = build_image_path(sample)
            mask_path = build_mask_path(sample)
            sample_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            sample_image_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            sample_image = apply_image_transformations(sample_image, sample_image_mask,
                    SEGMENT_IMAGE_WIDTH, SEGMENT_IMAGE_HEIGHT)
            
            image_output_path = image_output_directory / (mask_path.stem + ".png")
            # print(str(image_output_path))
            cv2.imwrite(str(image_output_path), sample_image)
            # Save the npy files so we can classify stuff afterwards
            # with image_output_path.open("wb") as fd:
            #     norm_image = cv2.normalize(sample_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #     np.save(fd, norm_image)

            datum_to_write = build_caffe_datum(sample_image, sample)
            lmdb_transaction.put(f"{sample_idx}".encode(), 
                    datum_to_write.SerializeToString())

    lmdb_db.close()

def process_and_store_images(segment_descriptors):
    image_output_directory = path.Path("./masked-images/")
    images_to_process = list(enumerate(segment_descriptors.values()))
    
    cs = defaultdict(int)
    for sample_idx, segment_descriptor in images_to_process:
        image_path = build_image_path(segment_descriptor)
        mask_path = build_mask_path(segment_descriptor)

        sample_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        sample_image_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        cs[sample_image.shape] += 1
        # print(sample_image.shape, sample_image_mask.shape)
        sample_image = cv2.bitwise_and(sample_image, sample_image,
                mask=sample_image_mask)
        sample_image = cv2.resize(sample_image, 
                (SEGMENT_IMAGE_WIDTH, SEGMENT_IMAGE_HEIGHT),
                interpolation=cv2.INTER_CUBIC)
        image_output_path = image_output_directory / (mask_path.stem + ".npy")
        if sample_idx % 200 == 0:
            with image_output_path.open("wb") as fd:
                np.save(fd, sample_image)
            # cv2.imwrite(str(image_output_path), sample_image)
            # Want to save numpy array files here

def print_lmdb_stats(lmdb_path):
    db = lmdb.open(str(lmdb_path), readonly=True)
    print(db.stat())


def build_caffe_datum(image, segment_descriptor):
    the_datum = caffe_pb2.Datum(
            channels=3,
            width=SEGMENT_IMAGE_WIDTH,
            height=SEGMENT_IMAGE_HEIGHT,
            label=segment_descriptor["data"]["segment_type"]["data"],
            data=image.tostring())

    return the_datum

def generate_image_mean_file(lmdb_path, output_path):
    os.system(f"compute_image_mean -backend=lmdb {str(lmdb_path)} {str(output_path)}")
            
def remove_old_databases(*args):
    for arg in args:
        os.system(f"rm -r {str(arg)}")

def generate_convert_imageset_file(segments):
    s = ""
    for segment_name, segment in segments.items():
        seg_path = path.Path(segment["Binary Segment Relative Path"])
        label = segment["data"]["segment_type"]["data"]
        s += f"{seg_path.name} {label}\n"
    s = s[:-1]
    return s

def oversample_data(segments_to_oversample):
    image_dir = path.Path("./masked-images")
    images = []
    ys = []
    for d in segments_to_oversample.values():
        image_path = str(image_dir / path.Path(d["Binary Segment Relative Path"]).name)
        the_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images.append(the_image)
        ys.append(d["data"]["segment_type"]["data"])

    images = np.array(images)
    print(images.shape)
    images = images.reshape(20000, 128 * 128 * 3)
    smt = SMOTE()
    x_train, y_train = smt.fit_sample(images, ys)
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 3)
    new_sd = {}
    for idx, im in x_train:
        new_sd[idx] = ("Mask_n_{idx}.png", y_train[idx]) 
    print(f"shape: {images.shape}")


def make_convert_imageset_files(segments_descriptor):
    label_dist = sorted(Counter([s_i["data"]["segment_type"]["data"] 
        for s_i in segments_descriptor.values()]).items(), key=op.itemgetter(0))
    # pp.pprint(label_dist)
    training_data, validation_data = split_data(segments_descriptor)
    training_txt = generate_convert_imageset_file(training_data)
    validation_txt = generate_convert_imageset_file(validation_data)
    path.Path("./training.txt").write_text(training_txt)
    path.Path("./validation.txt").write_text(validation_txt)

def augment_dataset(segments_descriptor):
    masked_path = path.Path("./masked-images")
    starting_index = max([int(s_name.split("_")[-1]) for s_name in segments_descriptor.keys()]) + 1
    keys_to_augment = list(segments_descriptor.keys())
    for key_value in keys_to_augment:
        segment = segments_descriptor[key_value]
        image_path = masked_path / path.Path(segment["Binary Segment Relative Path"]).name
        the_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        horizontal_flip = cv2.flip(the_image, 1)
        new_image_name = f"Mask_n_{starting_index}.png"
        cv2.imwrite(str(masked_path / new_image_name), horizontal_flip)
        new_segment = deepcopy(segment)
        new_segment["Binary Segment Relative Path"] = f"masks/Mask_n_{starting_index}.png"
        new_segment["Segment Relative Path"] = f"segments/Segment_n_{starting_index}.png"
        new_segment["is_flipped"] = 1
        segments_descriptor[f"Segment_n_{starting_index}"] = new_segment

        starting_index += 1

def opencfu_predict(image_path):
    out_file_path = path.Path(f"/tmp/outfile-{image_path.stem}.txt")
    opencfu_path = path.Path("/home/alexj/coursework/cs686/project/OpenCFU/opencfu")
    opencfu_predict_command = f"./opencfu -a -i {str(image_path.absolute())} > {str(out_file_path.absolute())}"
    os.system(opencfu_predict_command)
    result_txt = out_file_path.read_text().splitlines()
    count = len(result_txt) - 1
    return count

def generate_opencfu_data(segments_descriptor):
    training_data, validation_data = split_data(segments_descriptor)
    print("Validation data histogram")
    print(Counter([d_i["data"]["segment_type"]["data"] for d_i in segments_descriptor.values()]))

    results = {}
    masked_images_path = path.Path("./masked-images").absolute()
    os.chdir("/home/alexj/coursework/cs686/project/OpenCFU/")
    # for segment in itertools.islice(validation_data.values(), 10):
    for segment in validation_data.values():
        results_dict = {}
        segment_name = path.Path(segment["Binary Segment Relative Path"]).stem
        opencfu_prediction = opencfu_predict(masked_images_path / (segment_name + ".png"))
        results_dict["segment_name"]        = segment_name
        results_dict["label"]               = segment["data"]["segment_type"]["data"] 
        results_dict["text"]                = segment["data"]["segment_type"]["text"]
        results_dict["opencfu_prediction"]  = opencfu_prediction
        results[segment_name]               = results_dict

    os.chdir("/home/alexj/coursework/cs686/project/implementation/python-src/")
    return results

def undersample_dataset(segments):
    # # define undersample strategy
    # undersample = RandomUnderSampler(sampling_strategy='majority')
    # # fit and apply the transform
    # X_over, y_over = undersample.fit_resample(X, y)
    # # summarize class distribution
    # print(Counter(y_over))
    # names = [[s_i] for s_i in segments.keys()]
    labels = [s_i["data"]["segment_type"]["data"] for s_i in segments.values()]
    
    c0_count = len([v for v in labels if v == 0])
    print(f"Number of class 0 is {c0_count}")
    print(f"Total samples: {len(labels)}")
    new_segments = deepcopy(segments)
    random_of_class_1 = random.sample([k for k, d_i in segments.items() if d_i["data"]["segment_type"]["data"] == 0], 10000)
    for key_val in random_of_class_1:
        del new_segments[key_val]
    print(sorted(Counter([d_i["data"]["segment_type"]["data"] for d_i in new_segments.values()]).items()))
    print(f"Total samples after undersampling: {len(new_segments)}")
    return new_segments

def main():
    # segments_descriptor_file = DATASET_PATH / "enumeration-segments.json" 
    segments_descriptor_file = path.Path("./new-descriptor.json")
    
    segments_descriptor = json.loads(segments_descriptor_file.read_text())

    training_data, validation_data = split_data(segments_descriptor)
    undersampled_segments = undersample_dataset(training_data)
    convert_file = generate_convert_imageset_file(undersampled_segments)
    path.Path("segments-undersampled.json").write_text(json.dumps(undersampled_segments))
    path.Path("./training-undersampled.txt").write_text(convert_file)
    # print(convert_file)
    # path.Path("./training-undersampled.txt").write_text(convert_file)
    # results = generate_opencfu_data(segments_descriptor)
    # path.Path("./opencfu-results.json").write_text(json.dumps(results))
    # make_convert_imageset_files(segments_descriptor)
    # make_convert_imageset_files(undersampled_segments)
    # augment_dataset(segments_descriptor)
    # path.Path("new-descriptor.json").write_text(json.dumps(segments_descriptor))
    # print(len(segments_descriptor))
    # oversample_data(segments_descriptor)
    # print(f"Generating LMDB databases with {len(segments_descriptor)} total images.")
    # 
    # lmdb_databases = path.Path("../lmdb-databases")
    # training_set_lmdb_path = lmdb_databases / "training_set.lmdb"
    # validation_set_lmdb_path = lmdb_databases / "validation_set.lmdb"


    # # process_and_store_images(training_data)
    # os.system("rm -r ./masked-images")
    # os.system("mkdir masked-images")

    # if REGENERATE_LMDB_DATABASES:
    #     remove_old_databases(training_set_lmdb_path, validation_set_lmdb_path)
    #     build_lmdb(validation_set_lmdb_path, validation_data)
    #     build_lmdb(training_set_lmdb_path, training_data)
    #     generate_image_mean_file(training_set_lmdb_path, 
    #             lmdb_databases / "training-set-mean.binaryproto")

    # print_lmdb_stats(training_set_lmdb_path)
    # print("")
    # print_lmdb_stats(validation_set_lmdb_path)

if __name__ == "__main__":
    main()
