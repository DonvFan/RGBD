import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
from tools import *


def main(task_path:str):
    with open(os.path.join(task_path, 'log.txt'), '+a') as rf:
        rf.write('START!')
        out_dir = data_zip2ns(task_path)
        data_ingp2pl(out_dir)
        label_m2f(out_dir)
        label_m2f_pl(out_dir)
        o3d_tsdf_fusion(out_dir)
        run_dir = train_model(out_dir)
        inference_model(run_dir)
        rf.write('DONE!')
