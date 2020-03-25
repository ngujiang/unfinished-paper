"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
read_path = '/data/ljy/cycleGan-1-12/test/'
write_path = '/data/ljy/cycleGan-1-12/result6/'
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input' , read_path + "input1"  + ".jpg", 'input image path (.jpg)')
tf.flags.DEFINE_string('output' , write_path + "output1" + ".jpg", 'output image path (.jpg)')
tf.flags.DEFINE_string('model', '/data/ljy/pretrained/apple2orange.pb', 'model path (.pb)')

tf.flags.DEFINE_integer('image_size', '512', 'image size, default: 512')

def inference():
  graph = tf.Graph()
  for i in range(100):
    num=i+1
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    print(num)
    FLAGS.input = read_path + "input" + str(num) +".jpg"
    FLAGS.output=write_path +"output" + str(num) +".jpg"



    with graph.as_default():
      with tf.gfile.FastGFile(FLAGS.input, 'rb') as f:
        image_data = f.read()
        input_image = tf.image.decode_jpeg(image_data, channels=3)
        input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
        input_image = utils.convert2float(input_image)
        input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

      with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print(FLAGS.model)
        graph_def = tf.GraphDef()
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")

        graph_def.ParseFromString(model_file.read())
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")

      [output_image] = tf.import_graph_def(graph_def,
                            input_map={'input_image': input_image},
                            return_elements=['output_image:0'],
                            name='output')

    with tf.Session(graph=graph) as sess:
      generated = output_image.eval()
      with open(FLAGS.output, 'wb') as f:
        f.write(generated)


def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
