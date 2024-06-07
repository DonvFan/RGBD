# 1. 安装
完整运行环境详见[pip_list.txt](./pip_list.txt)和[conda_list.txt](./conda_list.txt)

## 1.1 PanopticField
* 安装MarchingCubes和TSDFFusion模块
```sh
cd envs/external/NumpyMarchingCubes
python setup.py install

cd envs/external/TSDFFusion
python setup.py install
```

* 依赖模块安装
```
pip install -r panopticlifting_re.txt
```

## 1.2 Nerfstudio
```
pip install nerfstudio
```
## 1.3 Mask2former
安装前需要使用`git submodule update`更新项目依赖，保证mask2former项目目录下。

安装mask2former参考[环境安装文档](./preprocess/mask2former/)，
详细使用mask2former方法参考[说明文档](./preprocess/mask2former/README.md)，预训练参数见[Model_Zoo](./preprocess/mask2former/MODEL_ZOO.md)文档，将参数下载至```preprocess/mask2former/checkpoints```文件夹下，推荐使用[model_final_f07440.pkl](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl)。


# 2. 运行
服务端多卡运行有爆内存风险，因此目前只支持单卡运行。运行命令：
```sh
CUDA_VISIBLE_DEVICES=0 python3 server.py
```
运行参数：
* `--port`: Flask App运行开放端口，默认12314
* `--data_root`: 上传数据存放文件夹，默认`./data`
* `--runs_root`: 运行结果存放文件夹，默认`./runs`

# 3. 程序接口简述
## 3.1 server.py
`server.py`是入口程序，负责处理客户端请求以及启动处理程序运行。

### 服务器状态查询
```python
@app.route('/rgbd2bim/serverstatus', methods = ['GET'])
def serverstatus()
```
返回服务器状态字典（JSON），
```json
{
    'Status': str, 
    'Waitlist':int
}
```
Status分为`RUNNING`（拥挤）、`FULL`（已满）、`EMPTY`（空闲）。
Waitlist为任务队列剩余名额数量。
### 任务上传
```python
@app.route('/rgbd2bim/upload', methods = ['GET', 'POST'])
def receive_client_file():
```
处理客户端新建任务请求，并接受polycam输出的原生压缩包(ZIP格式)，返回带有任务ID的字典：
```
{
    'id':str
}
```
客户端随后可通过任务id查询任务处理状态。polycam数据被存储在```<data_root>/<id>``` 文件夹下。

### 任务状态查询
```python
@app.route('/rgbd2bim/taskstatus/<string:taskid>', methods=['GET'])
def taskstatus(taskid):
```
查询对应ID任务的状态，返回任务状态字典
```json
{
    'Status': str, 
    'Waitlist':0
}
```
Status分为SUBMIT（提交）、RUNNING（处理中）、DONE（完成）三种状态。

### 处理结果下载
```python
@app.route('/rgbd2bim/taskresult/<string:taskid>', methods=['GET'])
def taskresult(taskid):
```
返回对应ID任务的重建结果，以ZIP压缩包的格式发送到客户端。返回结果主要包括命名为```labeled_pcd.txt```的点云文件，以“x y z label”格式存储点云位置和语义分类信息。语义分类标签遵守以下约定：
```
/// 1-floors 楼板
/// 2-roofs 屋顶
/// 3-beams 梁
/// 4-doors 门
/// 5-walls 墙
/// 6-wall openings 墙洞
/// 7-windows 窗
/// 8-curtain walls 幕墙
/// 9-columns 柱
/// 10-stairs 楼梯
/// 11-railings 栏杆
/// 12-furniture 家具
/// 13-lighting fixtures 灯具
/// 14-pipes 管道
/// 15-equipments 设备
/// 16-site 地形
/// 17-other 其它
```

### 消费者进程
```python
def consumer()
```
消费者进程从任务队列中弹出任务信息并运行主函数处理。

## 3.2 main.py
```
def main(task_path:str):
```




