"""Waymo 原始 TFRecord 到逐帧 pickle 的转换工具。

将 Waymo Open Dataset 的 tfrecord 逐帧解析为 lidar 点云与 annotation 两个
pickle 文件，供后续 infos 生成与训练数据加载使用。本模块不负责在线加载。

主要函数:
    - convert: 转换单个 tfrecord 文件内的所有帧。
    - main: 入口，负责 glob 文件列表并调度转换。

依赖 waymo_decoder 完成 frame 与 annotations 的解码。

改编自 https://github.com/WangYueFt/pillar-od (MIT License)。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, argparse, tqdm, pickle, os 

import waymo_decoder 
import tensorflow.compat.v2 as tf
from waymo_open_dataset import dataset_pb2

from multiprocessing import Pool 

tf.enable_v2_behavior()

fnames = None 
LIDAR_PATH = None
ANNO_PATH = None 

def convert(idx):
    """转换索引为 idx 的单个 tfrecord 文件中的所有帧。

    读取该 tfrecord 内每一帧，调用 waymo_decoder 解码出点云与标注，
    分别序列化为 lidar 与 annos 两个 pickle 文件（按 seq_{idx}_frame_{frame_id} 命名）。

    Args:
        idx (int): fnames 列表中的文件索引，同时用作输出文件名中的序列号。
    """
    global fnames
    fname = fnames[idx]
    dataset = tf.data.TFRecordDataset(fname, compression_type='')
    for frame_id, data in enumerate(dataset):
        # 逐条记录解析为 Frame protobuf
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        decoded_frame = waymo_decoder.decode_frame(frame, frame_id)
        decoded_annos = waymo_decoder.decode_annos(frame, frame_id)

        # 点云与标注分别落盘，文件名带离线排序用的序列号与帧号
        with open(os.path.join(LIDAR_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_frame, f)
        
        with open(os.path.join(ANNO_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_annos, f)


def main(args):
    """转换入口：展开 record_path 通配符得到文件列表并逐文件转换。

    Args:
        args: argparse 命名空间，需含 record_path 字段（支持 glob 通配符）。
    """
    global fnames
    fnames = sorted(list(glob.glob(args.record_path)))
    print("Number of files {}".format(len(fnames)))

    # debug：单进程顺序执行，避免 Pool 子进程
    for i in range(len(fnames)):
        convert(i)
    # with Pool(128) as p:  # change according to your cpu
    #     r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Data Converter')
    parser.add_argument('--root_path', type=str, required=False)
    parser.add_argument('--record_path', type=str, required=False)
    args.root_path = "data/Waymo"
    args.record_path = "dataset/waymo/tfrecord_training/*.tfrecord"

    args = parser.parse_args()


    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    LIDAR_PATH = os.path.join(args.root_path, 'lidar')
    ANNO_PATH = os.path.join(args.root_path, 'annos')

    if not os.path.isdir(LIDAR_PATH):
        os.mkdir(LIDAR_PATH)

    if not os.path.isdir(ANNO_PATH):
        os.mkdir(ANNO_PATH)
    
    main(args)
