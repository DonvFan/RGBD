from flask import Flask, request, send_file, abort 
import os
from pathlib import Path
import time
import argparse
from collections import deque
from queue import Queue
import shutil
import typing
import werkzeug

import multiprocessing as mp
app = Flask(__name__)
# resources = Queue() # Mutiple GPUs could be utilized later.
from main import main
current_tasks = Queue(8) #store task source dir path
source_pool = mp.Pool(1)

@app.route('/rgbd2bim/serverstatus', methods = ['GET'])
def serverstatus():
    status = "RUNNING"
    Waitlist = 0

    if current_tasks.empty():
        status = "EMPTY"
    elif current_tasks.full():
        status = "FULL"
        Waitlist = 1
    
    return {'Status':status, 'Waitlist':Waitlist}
    # return {'Status': "EMPTY", "Waitlist":0}

@app.route('/rgbd2bim/taskstatus/<string:taskid>', methods=['GET'])
def taskstatus(taskid):
    # statuscode = -3
    status = ""
    path = Path(runs_root) / taskid
    if path.exists():
        # statuscode += 1
        status = "SUBMIT"
        log_path = path / 'log.txt'
        if log_path.exists():
            status = "RUNNING"
            try:
                with open(log_path, 'r') as f:
                    last_line = f.readlines()[-1]
                if last_line.startswith('DONE'):
                    status = "DONE"
            except Exception:
                pass
            
    return {'Status': status, 'Waitlist':0}

@app.route('/rgbd2bim/taskresult/<string:taskid>', methods=['GET'])
def taskresult(taskid):
    root_path = Path(runs_root) / taskid
    result_path = root_path / 'results'
    if result_path.exists():
        zip_path = root_path / 'res.zip'
        if not zip_path.exists():
            shutil.make_archive(str(root_path / 'res'), format='zip', root_dir=root_path, base_dir='result')
        return send_file(str(zip_path), as_attachment=True, download_name='%s.zip'%taskid)
    return "Task not exists", 404

@app.route('/rgbd2bim/upload', methods = ['GET', 'POST'])
def receive_client_file():
    if request.method == 'GET':
        return 
    elif request.method == 'POST':
        # file = request.files['src']
        file = request.files['file']

        if file:
            filename = file.filename
            with open('test.txt', 'w') as f:
                f.write(filename)
            name = Path(filename).stem
            ext = Path(filename).suffix
            task_id = str(time.time_ns())
            task_id = task_id[::-1]
            save_dir = Path(data_root) / task_id
            save_dir.mkdir(parents=True)
            file.save(save_dir / ('source' + ext))
            current_tasks.put(str(save_dir))
            return {'id':task_id}
    return

# Only one GPU.
def consumer():
    while True:
        if current_tasks.empty():
            time.sleep(10.)
            continue
        else:
            path_str = current_tasks.get()
            source_pool.apply_async(consume, args=(path_str,))

def consume(path):
    main(path)


@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    return 'bad request!', 400

         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12314)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--data_root', type = str, default='./data')
    parser.add_argument('--runs_root', type = str, default='./runs')
    args = parser.parse_args()
    data_root = args.data_root
    runs_root = args.runs_root
    consumer_proc = mp.Process(target=consumer)
    consumer_proc.start()

    app.run(host=args.host, debug=True, port=args.port)
   
    # init_tasks(args.data_root)
    # app.run(host='0.0.0.0', debug=True, port=11411)
