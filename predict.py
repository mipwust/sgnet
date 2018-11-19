import tensorflow as tf
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse


def decode_labels(image, label):
    """ store label data to colored images """
    layer1 = [255, 0, 0]
    layer2 = [255, 165, 0]
    layer3 = [255, 255, 0]
    layer4 = [0, 255, 0]
    layer5 = [0, 127, 255]
    layer6 = [0, 0, 255]
    layer7 = [127, 255, 212]
    layer8 = [139, 0, 255]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8])
    for l in range(0, 7):
        r[label == l+1] = label_colours[l, 0]
        g[label == l+1] = label_colours[l, 1]
        b[label == l+1] = label_colours[l, 2]
    r[label == 9] = label_colours[7, 0]
    g[label == 9] = label_colours[7, 1]
    b[label == 9] = label_colours[7, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r/255.0
    rgb[:, :, 1] = g/255.0
    rgb[:, :, 2] = b/255.0
    return rgb


def predict(checkpoint_dir, image_file):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        new_saver.restore(sess, latest_checkpoint)
        graph = tf.get_default_graph()
        pred_tensor = tf.get_collection('prediction')[0]
        image_placeholder = graph.get_operation_by_name('image').outputs[0]
        is_training = graph.get_tensor_by_name('Placeholder:0')[0]

        image_path = os.path.join('images', image_file)
        image = io.imread(image_path, as_gray=True)
        image = image[np.newaxis, ..., np.newaxis]
        image = image / 127.5 - 1.0

        fetches = [pred_tensor]
        feed_dict = {image_placeholder: image, is_training: False}
        preds = sess.run(fetches, feed_dict=feed_dict)

        preds = np.squeeze(preds)
        image = np.squeeze(image)
        image = np.squeeze((image + 1) * 127.5)

        labeled_image = decode_labels(image, preds)
        plt.figure(1)
        ax1 = plt.subplot(121)
        plt.imshow(np.uint8(image), cmap='gray')
        ax1.set_axis_off()
        ax2 = plt.subplot(122)
        plt.imshow(labeled_image)
        ax2.set_axis_off()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--save_path', '-s', type=str, default='outputs')
    parser.add_argument('--checkpoint_dir', '-c', type=str, default='model')
    parser.add_argument('--image_file', '-i', type=str, default='image_1.png')
    args = parser.parse_args()

    wd = os.path.dirname(os.path.realpath(__file__))
    args.save_path = os.path.join(wd, args.save_path)
    args.checkpoint_dir = os.path.join(wd, args.checkpoint_dir)

    predict(args.checkpoint_dir, args.image_file)
