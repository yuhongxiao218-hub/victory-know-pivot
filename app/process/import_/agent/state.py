import copy
from typing import TypedDict, Any



class ImportGraphState(TypedDict):
    #任务标识
    task_id: str

    #输入文件和输出文件夹
    local_file_path: str #输入文件夹地址
    local_dir:str #输出文件夹地址

    #标记文件类型
    is_md_read_enabled:bool #是不是md
    is_pdf_read_enabled:bool #是不是pdf

    #存储对应类型的文件地址
    md_path:str #专门存储md文件地址，前身是local_file_path
    pdf_path:str #专门存储pdf文件地址

    #存储数据内容
    md_content:str #md文件内容，后续用于切割
    item_name:str #文件唯一标识符
    file_title:str #文件标题 去掉后缀名 xx.txt->xx

    #切片后的数据
    chunks:list[dict[str,Any]] #切片内容 字典里{content:x,item_name:x}  未向量化的
    embeddings_content:list[dict[str,Any]] #装生成向量的切片内容 稠密和稀疏


#定义一个对象模板，可以赋予少量的默认值
graph_default_state: ImportGraphState = {
        "task_id": "",
        "is_pdf_read_enabled": False,
        "is_md_read_enabled": False,
        "local_dir": "",
        "local_file_path": "",
        "pdf_path": "",
        "md_path": "",
        "file_title": "",
        "md_content": "",
        "chunks": [],
        "item_name": "",
        "embeddings_content": [],
    }

#创建两个快速创建对象的方法
"""
copy
import copy
深拷贝：copy.deepcopy(模板对象)  # 深拷贝，复制对象及其引用的对象也就是嵌套属性
浅拷贝：copy.copy(obj)  # 浅拷贝，只复制对象
"""
#**kwargs:传好多key:value用kwargs（本质是字典）
def create_default_state(**kwargs)->ImportGraphState:
        """
        快速创建一个指定属性的state对象，会覆盖默认值
        """
        new_state:ImportGraphState=copy.deepcopy(graph_default_state)
        # for key, value in kwargs.items():
        #     if key in new_state:
        #         new_state[key] = value
        new_state.update(kwargs)
        return new_state
def get_default_state()->ImportGraphState:
        """
        快速返回模板对象，不修改数据
        """
        new_state:ImportGraphState=copy.deepcopy(graph_default_state)
        return new_state

