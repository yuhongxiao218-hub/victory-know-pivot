import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, BackgroundTasks, UploadFile
from filetype import guess_mime
from pymilvus.grpc_gen.common_pb2 import ImportState

from app.api.schema.import_schema import UploadResponse, StatusResponse
from app.process.import_.agent.main_graph import import_graph_app
from app.process.import_.agent.state import create_default_state, ImportGraphState
from app.shared.runtime.logger import logger
from app.shared.utils.path_util import PROJECT_ROOT
from app.shared.utils.task_utils import get_task_status, get_running_task_list, get_done_task_list, update_task_status, \
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

app=FastAPI(description="导入接口模版")

@app.get("/html")
def return_import_html():
    html_file_obj:Path=PROJECT_ROOT / "app" /"resources" / "html" / "import.html"
    return FileResponse(
        path=str(html_file_obj),
       # media_type=guess_mime(html_file_obj.name)[0]
        media_type = "text/html; charset=utf-8"
    )

def invoke_import_graph(task_id:str,local_file_path:Path,local_dir:str):
    state:ImportState=create_default_state(
        task_id=task_id,
        local_file_path=local_file_path,
        local_dir=local_dir
    )
    logger.info(f"执行参数：\n{json.dumps(state,indent=4,ensure_ascii=False)}")
    try:
        #开始状态
       update_task_status(task_id,TASK_STATUS_PROCESSING)
       result:ImportGraphState=import_graph_app.invoke(state)
        #成功状态
       update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        logger.exception(f"导入流程发生了异常：{state}")
        #失败状态
        update_task_status(task_id, TASK_STATUS_FAILED)
@app.post("/upload")
def upload_file(tasks:BackgroundTasks, files:list[UploadFile]):
    # 1.接收文件
    # 固定前端就会传递一个文件,但是用列表包装了
    current_file: UploadFile = files[0]
    # 2.文件转存
    # 存到哪? /output/20261010/task_id -> local_dir  /filename -> local_file_path
    # 2.1 task_id
    task_id = str(uuid.uuid4())
    # 2.2 local_dir
    local_dir_obj: Path = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    # parents 创建多层文件夹
    # exist_ok 存在不报错 不创建
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    # 2.3 local_file_path
    local_file_path_obj: Path = local_dir_obj / current_file.filename
    # 2.4 转存文件
    # 方案1: 一次性读取,直接一次性写入 [简单/沾满内存]
    #todo:await是因为read()是异步的  那么def也要加asyc
    # data = await current_file.read()
    # local_file_path_obj.write_bytes(data)


    with local_file_path_obj.open("wb") as output_file:#写到output_file
        # input_file = current_file.file

        # shutil  移除文件夹 rmtree 压缩 unpack_archive(压缩文件,解压的文件夹(地点))  解压缩 make_archive("压缩后的文件名","压缩格式zip/tar","压缩的文件夹")
        # 逐步的读取输入的流的数据,写到输出流中
        # 参数1: 输入流 从哪个流读取
        # 参数2: 输出流 写入到哪个位置
        # COPY_BUFSIZE = 1024 * 1024 if _WINDOWS else 64 * 1024
        shutil.copyfileobj(current_file.file, output_file)


    # 3.异步调用图对象的执行
    # 异步: 1. 提取一个独立的方法 2.tasks.add_task(函数,...)
    tasks.add_task(invoke_import_graph, task_id=task_id, local_file_path=str(local_file_path_obj),
                   local_dir=str(local_dir_obj))

    # 4.返回结果即可
    return UploadResponse(
        code=200,
        message=f"{current_file.filename}已经完成上传，正在解析中",
        task_ids=[task_id],

    )


# 1. 获取html
# 2. 上传文件 -> langgraph的执行
# 3. 根据文件task_id -> 获取解析的过程
"""
   2. 上传文件 -> langgraph的执行 -> 1 2 3 4 5 6 7
                                       |
                                 dict{task_id:[]}  
                                       ^ 
                                       |
   3. 根据文件task_id -> 获取解析的过程---->
"""

"""

   前端 -> 后端python业务

   前端 <- 后端python前端
      轮询: 并不是后端推送到前端  最简单/性能最差[空请求]   前端主动
      sse: 真.后端向前端推送数据技术        后端主动    单向  前 <-  后
      websocket: 真.后端向前端推送数据技术  后端主动    双向  前 <-> 后
"""


# 接口3  2秒一次
@app.get("/status/{task_id}")
def statsu_info(task_id: str):
    # todo task_id相关的进度 1. 任务状态  2. 进行中节点列表 3. 完成的节点列表
    # task_util.py 容器工具类 [存储空间 dict] [存储数据方法] [读取数据方法]
    """
      _tasks_running_list: Dict[str, List[str]] = {}
           add_running_task         -> 每个节点存储完毕
           get_running_task         -> 接口
      _tasks_done_list: Dict[str, List[str]] = {}
           add_done_task            -> 每个节点存储完毕
           get_done_task            -> 接口
      _tasks_status: Dict[str, str] = {}
           update_task_status       -> 执行langgraph的try代码块中
           get_task_status          -> 接口
    """
    status = get_task_status(task_id)
    running_list = get_running_task_list(task_id)
    done_list = get_done_task_list(task_id)

    return StatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        running_list=running_list,
        done_list=done_list
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
