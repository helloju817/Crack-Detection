import os
import glob
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import SR_models
import utils
import cv2
import time

tf.app.flags.DEFINE_string('model_path', './modelsavedPAT', '/where/your/model/folder')
# tf.app.flags.DEFINE_string('image_path', '/where/your/test_image/folder', '')
#tf.app.flags.DEFINE_string('save_path', './Train/train_lab', '/where/your/generated_image/folder')
tf.app.flags.DEFINE_string('save_path', 'C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack4/cnn(affine_60)2', '/where/your/generated_image/folder')
tf.app.flags.DEFINE_string('run_gpu', '1', '')
#tf.app.flags.DEFINE_string('pwd','./Train/train___0816', '/where/your/pwd')
#pwd : test image path
tf.app.flags.DEFINE_string('pwd','C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack4/elastic(affine_60)2', '/where/your/pwd')
tf.app.flags.DEFINE_string('output','./Output', '/where/your/output_result')
tf.app.flags.DEFINE_float('SR_scale', 8, '')
FLAGS = tf.app.flags.FLAGS

pwd = FLAGS.pwd

def load_model(model_path):
    '''
    model_path = '.../where/your/save/model/folder'
    '''
    input_low_images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input_low_images')

    model_builder = SR_models.model_builder()

    generated_high_images, resized_low_images = model_builder.generator(input_low_images, is_training=False, model='enhancenet')

    generated_high_images = tf.cast(tf.clip_by_value(generated_high_images, 0, 255), tf.float32)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    gen_vars = [var for var in all_vars if var.name.startswith('generator')]

    saver = tf.train.Saver(gen_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_path = utils.get_last_ckpt_path(model_path)
    saver.restore(sess, ckpt_path)

    return input_low_images, generated_high_images, sess

def real_num_r(data,width,height,num, domain_size=""):
    data=np.array(data,dtype="float32")
    data = data / 255.0
    num_st = str(num)
    if (0 <= num) and (num < 10):
        new_num = "00" + num_st
    elif (10 <= num) and (num < 100):
        new_num = "0" + num_st
    else:
        new_num = num_st

    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    file = open(FLAGS.output + '/geometry'+str(width)+"x"+str(height)+'(X2)-' + new_num + '.txt','w')
    # print(data.shape)

    w_h = str(width)+" "+ str(height)+"\n"
    #file.write(domain_size)
    file.write(w_h)
    for x in range(width):
        for y in range(height):
            write_rgb = "{0} {1}".format(x,y)
            bgr = " {0} {1} {2}\n".format(data[y][x][2], data[y][x][1], data[y][x][0])
            file.write(write_rgb+bgr)
    file.close()

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank imag
    image = np.zeros((height, width, 3), np.float64)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def Test_Image_Processing_txt():
    contents=[]
    for path, dirs, files in os.walk(pwd):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.txt':
                asd = pwd + '/' + file
                filein = open(asd, 'r')
                #filein.seek(7)
                domain_size=""
                #for i in range(2):
                #    domain_size += filein.readline()
                line = filein.readline()
                line = line.split()
                width,height = int(line[0]),int(line[1])
                # image = create_blank(width, height, (0, 0, 0))
                # image = np.array(image)
                image = np.zeros([height, width, 3])
                while True:
                    line = filein.readline()
                    if not line:
                        break
                    line = line.split()
                    x = int(line[0])  # col
                    y = int(line[1])  # row
                    if x != width and y != height:
                        for i in range(3):
                            image[y, x, i] = (float(line[2 + i]) * 255.0)
                # while True:
                #     line = filein.readline()
                #     if not line:
                #         break
                #     line = line.split()
                #     x = int(line[0])    #col
                #     y = int(line[1])   #row
                #     if x != width and y != height:
                #         image[y][x] = (float(line[2]), float(line[3]), float(line[4]))  #bgr
                # print(file)
                filein.close()
                #cv2.imwrite('randomRGB.jpg', image)
                # data = np.array(image, dtype="float32")
                # data = data * 255.0
                # data = np.clip(data, 0, 255.0)
                contents.append(image)
    #return contents,domain_size
    return contents

def Test_Image_Processing():
    # dirname = []
    contents= []
    # file_count = []
    file_name = []
    # for path, dirs, files in os.walk(pwd):
    #     if len(dirs) > 0:
    #         for dir_name in dirs:
    #             if os.path.splitext(dir_name)[1].lower() != '.txt':
    #                 dirname.append(dir_name)
    #     if os.path.isdir(path) and len(files) > 0:
    #         print("dir_name:" + path)
    #         if os.path.splitext(path)[1].lower() != '.txt':
    #         #     file_count.append(len(files) / 2)
    #         #     print("file_count: " + str(len(files)/2))
    #             i = 0
    #             for file in files:
    #                 if os.path.splitext(file)[1].lower() == '.jpg':
    #                     asd = os.path.join(path, file)
    #                     #print(asd)
    #                     image = cv2.imread(asd)
    #                     contents.append(image)
    #                     file_name.append(file)
    #                     i += 1
    #             file_count.append(i)
    #     #print(path + ": " + str(contents.__len__()))
    # return contents, dirname, file_count, file_name
    #return contents, file_name
    for path, dirs, files in os.walk(pwd):
        for file in files:
            asd = os.path.join(path, file)
            image = cv2.imread(asd)
            image = cv2.resize(image, (0, 0), fx=1/FLAGS.SR_scale, fy=1/FLAGS.SR_scale)
            contents.append(image)
            file_name.append(file)
    print(path + ": " + str(contents.__len__()))
    return contents, file_name


def init_resize_image(im):
    h, w, _ = im.shape
    size = [h, w]
    max_arg = np.argmax(size)
    max_len = size[max_arg]
    min_arg = max_arg - 1
    min_len = size[min_arg]

    maximum_size = 1024
    if max_len < maximum_size:
        maximum_size = max_len
        ratio = 1.0
        return im, ratio
    else:
        ratio = maximum_size / max_len
        max_len = max_len * ratio
        min_len = min_len * ratio
        size[max_arg] = int(max_len)
        size[min_arg] = int(min_len)

        im = cv2.resize(im, (size[1], size[0]))

        return im, ratio


start = time.time()  # start time save
print("START")
if __name__ == '__main__':
    # set your gpus usage
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.run_gpu

    # get pre-trained generator model
    input_image, generated_image, sess = load_model(FLAGS.model_path)

    # get test_image_list
    # test_image_list = utils.get_image_paths(FLAGS.image_path)
    #test_image_list = Test_Image_Processing_txt()
    #test_image_list, dir_names, fc, fn = Test_Image_Processing()
    test_image_list , fn = Test_Image_Processing()
    #test_image_list, domain_size = Test_Image_Processing_txt()
    # make save_folder

    # if not os.path.exists(FLAGS.save_path):
    #     os.makedirs(FLAGS.save_path)

    # do test
    i = 0
    num = 0
    for test_idx, test_image in enumerate(test_image_list):

        tmp_save_path = FLAGS.save_path

        loaded_image = test_image
        processed_image, tmp_ratio = init_resize_image(loaded_image)

        feed_dict = {input_image : [processed_image[:,:,::-1]]}

        output_image = sess.run(generated_image, feed_dict=feed_dict)

        output_image = output_image[0,:,:,:]

        image_name = fn[test_idx]
        h, w, _ = output_image.shape
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(FLAGS.save_path + "/" + image_name , output_image)
        cv2.imwrite(tmp_save_path + '/' + image_name, output_image)
        #real_num_r(output_image, w, h, test_idx, domain_size)
        #real_num_r(output_image, w, h, test_idx)
        #plt.imsave(tmp_save_path, output_image)


        print("%d %d"%(w,h))
        print('%d / %d completed!!!' % (test_idx + 1, len(test_image_list)))
print("END")
print("time :", time.time() - start)





