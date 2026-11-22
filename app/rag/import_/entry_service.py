from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger
from pathlib import Path


def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
   """
   「校验 + 按类型分流 + 提取文件名」的入口归一化函数；而且它的「校验」只到非空和后缀为止，不校验文件是否存在和可读。
   :param state: 导入流程状态，需包含 local_file_path（待处理文件的地址）
   :return: 更新后的 state（回写 is_md_read_enabled / is_pdf_read_enabled / md_path / pdf_path / file_title）
   """
   # 1. 先从state读取使用的数据 local_file_path  task_id  注意: 别要直接 [] 避免KeyError 使用get...
   local_file_path:str=state.get("local_file_path")
   # 2. local_file_path做非空判断  注意: 必须判断,不要过度!
   if not local_file_path:
       #为空 要raise
       logger.error(f"local_file_path为空，业务无法进行，提前终止！！")
       raise ValueError(f"local_file_path为空，业务无法进行，提前终止！！")
   # 3. 不为空进行后缀名判断 md / pdf -> 修改对应的值 else 既不是md又不是pdf 打印日志..
   if local_file_path.lower().endswith(".md"):
       state['is_md_read_enabled']=True
       state['is_pdf_read_enabled']=False
       state['md_path']=local_file_path
   elif local_file_path.lower().endswith(".pdf"):
       state['is_md_read_enabled'] = False
       state['is_pdf_read_enabled'] =True
       state['pdf_path'] = local_file_path
   else:
       return  state
   # 4. 将地址转成Path对象 (os.path) 快速获取文件file_title
     #(1)os.path.basename()/os.path.splitext  3.4python之前
     #(2)之后使用pathlib包下的Path进行面向对象思路进行地址操作
   local_file_obj:Path=Path(local_file_path)
   file_title:str     =local_file_obj.stem
   """
        path使用分三步:
        1.创建Path对象
                  x:Path=Path(字符串地址)     获取地址对应的Path对象
                  x:Path=Path(Path对象)
                  x:Path=Path()             获取当前文件所在的文件夹
                  x:Path=Path(__file__)     获取当前文件(写代码的这个文件)的path地址
                  x:Path=path对象 /"字符串地址" /"字符串地址"
        2. .属性
                .name    文件的完整文件名  info.pdf
                .stem    获取文件名部分    info
                .suffix  获取后缀名部分    .pdf
                .parent  获取父Path对象   Path(__file__).parent.parent
                .parents 获取所有的父文件夹 Path(__file__).parents[1] = [import_,rag,app,项目根地址..]
        3. .方法()  创建一个文件夹 查询文件是否存在
   """
   # 5. state属性更新
   state['file_title'] = file_title
   # 6. 返回state
   return state



    
