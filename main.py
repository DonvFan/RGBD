import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
from tools import *


def main(task_path:str):
    with open(os.path.join(task_path, 'log.txt'), '+a') as rf:
        rf.write('START!')
    out_dir = data_zip2ns(task_path)
    data_ingp2pl(out_dir)
    label_m2f(out_dir)
    label_m2f_pl(out_dir)
    # # get_tsdf_grid(out_dir)
    o3d_tsdf_fusion(out_dir)
    run_dir = train_model(out_dir , 1)
    inference_model(run_dir)
    with open(os.path.join(task_path, 'log.txt'), '+a') as rf:
        rf.write('DONE!')


if __name__ == "__main__":
    p = '/home/fgm/disk1/Focus/code/system/data/test'
    main(p)