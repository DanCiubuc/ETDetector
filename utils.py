import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

SPLIT_RATIO = 0.33
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0

def convert_coordinates(coord, image_width = 1024, image_height = 768):
    # Convert normalized XYXY to pixel values
    x1_pixel =coord[0] * image_width
    y1_pixel =coord[1] * image_height
    x2_pixel =coord[2] * image_width
    y2_pixel =coord[3] * image_height
    # Convert normalized XYWH to pixel values
    x_center =coord[0] * image_width
    y_center =coord[1] * image_height
    width_pixel =coord[2] * image_width
    height_pixel =coord[3] * image_height
    x1_pixel = x_center - width_pixel / 2
    y1_pixel = y_center - height_pixel / 2
    x2_pixel = x_center + width_pixel / 2
    y2_pixel = y_center + height_pixel / 2
    return [x1_pixel, y1_pixel, x2_pixel, y2_pixel]

def parse_txt(txt_file, grayscale):
    width = 1280
    height = 720
    if grayscale:
         width = 1024 
         height = 768
         
    image_path = os.path.splitext(txt_file)[0] + '.png'

    boxes = []
    classes = []
    for line in open(txt_file):
        fields = line.split()
        label = float(fields[0])
        xmin = float(fields[1])
        ymin = float(fields[2])
        xmax = float(fields[3])
        ymax = float(fields[4])
        boxes.append(convert_coordinates([xmin, ymin, xmax, ymax], image_width = width, image_height = height))
        classes.append(label)


    return image_path, boxes, classes

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Use channels=3 for RGB images
    if image.shape.as_list()[-1] == 1:
        image = tf.concat([image, image, image], axis=-1)
        print('hello')
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs
    print(bounding_boxes['boxes'])
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        line_thickness=0,
        y_true=bounding_boxes,
        font_scale = 0.5,
        scale=10,
        bounding_box_format=bounding_box_format,
        # class_mapping=class_mapping,
    )
    
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs

def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        # class_mapping=class_mapping,
    )

def prepare_dataset(path,split_ds = True, augmenter = None, grayscale = None):
    # Get all TXT file paths in path_annot and sort them
    txt_files = sorted(
        [
            os.path.join(path, file_name)
            for file_name in os.listdir(path)
            if file_name.endswith(".txt")
        ]
    )

    # Get all PNG image file paths in path_images and sort them
    png_files = sorted(
        [
            os.path.join(path, file_name)
            for file_name in os.listdir(path)
            if file_name.endswith(".png")
        ]
    )
    
    it = os.scandir(path)

    image_paths = []
    bbox = []
    classes = []

    for entry in it:
        if entry.name.endswith('.txt'):
            image_path, boxes, class_ids = parse_txt(entry.path, grayscale)
            image_paths.append(image_path)
            bbox.append(boxes)
            classes.append(class_ids)
    
    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)

    data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
     
    if augmenter == None:
        augmenter = keras.Sequential(
        layers=[
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
        ),
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
        keras_cv.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
        )
        ])
        
        augmenter_val = keras.Sequential(
        layers=[
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
        )])
        
    dataset = None
    
    if split_ds is True:
        # Determine the number of validation samples
        num_val = int(len(txt_files) * SPLIT_RATIO)

        # Split the dataset into train and validation sets
        val_data = data.take(num_val)
        print(len(val_data))
        train_data = data.skip(num_val)
        print(len(train_data))
        
        
        train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(BATCH_SIZE * 4)
        train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
        print(train_ds)
        train_ds = train_ds.map(augmenter_val, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        # TODO: fazer augmenter separado para valdiation dataset
        val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.shuffle(BATCH_SIZE * 4)
        val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
        val_ds = val_ds.map(augmenter_val, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        
        dataset = (train_ds, val_ds)
    else:
        dataset = data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(BATCH_SIZE * 4)
        dataset = dataset.ragged_batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
    
