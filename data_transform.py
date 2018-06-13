import numpy as np
import os
import re
import tensorflow as tf
def flatten(x):
    x = np.array(x)
    shape = x.shape
    fla = 1
    for i in shape:
        fla*=i
    return np.reshape(x,(1,fla)),shape

def weights_flatten_and_save_txt(path = 'layers_weighs'):
    print(os.path.exists(path))
    weights_list = os.listdir(path)
    print(weights_list)
    for i,weights_path in enumerate(weights_list):
        w_path = os.path.join(path,weights_path)
        w_yuan = np.load(w_path)
        w_yuan = np.clip(w_yuan,0,1)
        print(w_yuan.shape)
        w,shape= flatten(w_yuan)
        print(w.shape)
        w.astype(np.int)
        if not os.path.exists('layer_weights_txt'):
            os.makedirs('layer_weights_txt')
        newname = os.path.join('layer_weights_txt',re.sub('npy','txt',weights_path))
        np.savetxt(newname,w,fmt='%d')
        print('save txt complete!')
        a = np.loadtxt(newname,dtype=int)
        print(a)
        a = np.reshape(a,shape)
        if np.sum(a-w_yuan)==0:
            print('txt data right!')
        else:
            raise('data wrong!')

def activites_flatten_and_save_txt(path = 'layers_out'):
    print(os.path.exists(path))
    weights_list = os.listdir(path)
    print(weights_list)
    if not os.path.exists('layers_out_txt'):
        os.makedirs('layers_out_txt')
    for i, weights_path in enumerate(weights_list):
        w_path = os.path.join(path, weights_path)
        w_yuan = np.load(w_path)
        print(w_yuan.shape)
        w, shape = flatten(w_yuan)
        print(w.shape)
        w.astype(np.int)
        newname = os.path.join('layers_out_txt', re.sub('npy', 'txt', weights_path))
        np.savetxt(newname, w, fmt='%d')
        print('save txt complete!')
        a = np.loadtxt(newname, dtype=int)
        print(a)
        a = np.reshape(a, shape)
        if np.sum(a - w_yuan) == 0:
            print('txt data right!')
        else:
            raise ('data wrong!')
def activites_flatten_and_save_txt_bin(path = 'layers_out'):
    print(os.path.exists(path))
    weights_list = os.listdir(path)
    print(weights_list)
    if not os.path.exists('layers_bin_out_txt'):
        os.makedirs('layers_bin_out_txt')
    for i, weights_path in enumerate(weights_list):
        w_path = os.path.join(path, weights_path)
        w_yuan = np.load(w_path)
        print(w_yuan.shape)
        w, shape = flatten(w_yuan)
        print(w.shape)
        #w.astype(np.uint8)
        w = np.unpackbits(np.array(w,dtype=np.uint8),axis=0)
        #w = w.T
        print(w.shape)

        newname = os.path.join('layers_bin_out_txt', re.sub('npy', 'txt', weights_path))
        np.savetxt(newname, w, fmt='%d')
        print('save txt complete!')
def activites_bn_flatten_and_save_txt(path = 'layers_bn_out'):
    print(os.path.exists(path))
    weights_list = os.listdir(path)
    print(weights_list)
    if not os.path.exists('layers_bn_out_txt'):
        os.makedirs('layers_bn_out_txt')
    for i, weights_path in enumerate(weights_list):
        w_path = os.path.join(path, weights_path)
        w_yuan = np.load(w_path)
        print(w_yuan.shape)
        w, shape = flatten(w_yuan)
        print(w.shape)
        w.astype(np.float32)
        newname = os.path.join('layers_bn_out_txt', re.sub('npy', 'txt', weights_path))
        np.savetxt(newname, w, fmt='%0.8f')
        print('save txt complete!')
        a = np.loadtxt(newname, dtype=float)
        print(a)
        a = np.reshape(a, shape)
        if np.abs(np.sum(a - w_yuan)) < 1e-5:
            print('txt data right!')
        else:
            raise ('data wrong!')
def bn_flatten_and_save_txt(path = 'bn_para'):
    print(os.path.exists(path))
    weights_list = os.listdir(path)
    print(weights_list)
    if not os.path.exists('bn_para_txt'):
        os.makedirs('bn_para_txt')
    for i, weights_path in enumerate(weights_list):
        w_path = os.path.join(path, weights_path)
        w_yuan = np.load(w_path)
        print(w_yuan.shape)
        w, shape = flatten(w_yuan)
        print(w.shape)
        w.astype(np.float32)
        newname = os.path.join('bn_para_txt', re.sub('npy', 'txt', weights_path))
        np.savetxt(newname, w, fmt='%0.8f')
        print('save txt complete!')
        a = np.loadtxt(newname, dtype=float)
        print(a)
        a = np.reshape(a, shape)
        if np.abs(np.sum(a - w_yuan)) < 1e-5:
            print('txt data right!')
        else:
            raise ('data wrong!')

# bn_flatten_and_save_txt()
# path = 'layers_weighs'
# print(os.path.exists(path))
# weights_list = os.listdir(path)
# print(weights_list)
# w_path = os.path.join(path,weights_list[0])
# w_yuan = np.load(w_path)
# w_yuan = np.clip(w_yuan,0,1)
# print(w_yuan.shape)

# newname = os.path.join('layer_weights_txt',re.sub('npy','txt',weights_list[0]))
# np.savetxt(newname,w,fmt='%d')
# print('save txt complete!')
# a = np.loadtxt(newname,dtype=int)
# print(a)
# a = np.reshape(a,shape)
# if np.sum(a-w_yuan)==0:
#     print('txt data right!')
# else:
#     raise('data wrong!')

def test_tf_conv():
    conv_w = tf.Variable([[[[1,5]],
                         [[2,6]]],
                        [[[3,7]],
                         [[4,8]]]
                        ],dtype=tf.float32)
    map = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]],dtype=np.float32)
    map_t = tf.convert_to_tensor(map)
    conv = tf.nn.conv2d(map_t,conv_w,[1,1,1,1],'VALID')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        wei = sess.run(conv_w)
        m = sess.run(map_t)
        c = sess.run(conv)
        print(wei)
        print(wei.shape)
        print(m)
        print(m.shape)
        print(c)
        print(c.shape)

        w, shape = flatten(wei)
        print(w.shape)
        w.astype(np.int)
        np.savetxt('test.txt', w, fmt='%d')
        print('save txt complete!')
        a = np.loadtxt('test.txt', dtype=int)
        print(a)
        a = np.reshape(a, shape)
        print(a)

        if np.sum(a - wei) == 0:
            print('txt data right!')
        else:
            raise ('data wrong!')

test_tf_conv()