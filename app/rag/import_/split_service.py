import json
import re

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
validates_and_get_data(state:ImportGraphState) -> tuple[str,str,Path]
    1.1 获取相关参数 md_content / file_title /  md_path 
    1.2 非空校验 
        md_content -> 空 -> 利用 -> md_path 二次读取 -> 还空 -> 抛异常
        file_title -> 空 -> 利用 -> md_path 二次读取 -> 还空 -> default 
    1.3 返回数据结果即可

"""
def validates_and_get_data(state:ImportGraphState) -> tuple[str,str,Path]:
    # 1.1 获取相关参数 md_content / file_title /  md_path
    md_content:str = state.get("md_content")
    md_path   :str = state.get("md_path")
    file_title:str = state.get("file_title")
    # 1.2 非空校验
    #校验「md 文件地址是否存在」
    if not md_path:
        logger.error(f"md_path执行的md文件为空,业务无法继续,提前终止!!")
        raise ValueError(f"md_path执行的md文件为空,业务无法继续,提前终止!!")
    md_path_obj:Path = Path(md_path)
    #校验「这个地址是不是真有个文件」
    if not md_path_obj.is_file():
        logger.error(f"md_path地址为:{md_path},但是没有对应的文件!业务无法继续,提前终止!!")
        raise FileNotFoundError(f"md_path地址为:{md_path},但是没有对应的文件!业务无法继续,提前终止!!")
    #校验「正文内容」，空了用 md_path 读文件兜底
    if not md_content:
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"md_content为空,读取md_path依然为空,业务无法继续,提前终止!!")
            raise ValueError(f"md_content为空,读取md_path依然为空,业务无法继续,提前终止!!")
        state['md_content'] = md_content
    #校验「文件标题」，空了用 md_path 读文件兜底
    if not file_title:
        file_title = md_path_obj.stem
        logger.warning(f"file_title为空,给与默认值:{file_title},业务继续!")
    # 1.3 对md_content的换行符号进行统一处理(数据清洗)
    # 替换所有换行符为 \n，解决 Windows/Linux 换行符不一致问题
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 1.4 返回对应的结果
    return md_content,file_title,md_path_obj


"""
 document_split_by_title(md_content:str) -> list[dict[str,Any]]:  
        2.1 先创建存储数据容器 [ 1.存储历史标题的数据 list[dict[str,Any]]   2. current_title:str  3. current_title_lines:list[str] ]
        2.2 对清洗后的md_content进行按行切割 \n -> list[str]
        2.3 定义正则匹配对应的标题
        2.4 循环md_content切割的行 逐行处理[标题/普通行]
            2.5 查看是不是标题
            2.6 是-> 结算上一赛季 / 记录新的赛季
            2.7 不是 -> 普通行 -> 追加到 current_title_lines 继续下一行处理即可
        2.8 跳出循环,最后进行数据结算 [因为最后一个标题没有人触发结束]
        2.9 返回结果

"""

# 原则一: 语义优先
def document_split_by_title(md_content:str) -> list[dict[str,Any]]:
    # 2.1 先创建存储数据容器 [ 1.存储历史标题的数据 list[dict[str,Any]]   2. current_title:str  3. current_title_lines:list[str] ]
    chunks_title:list[dict[str,Any]] = []#最终结果，装所有切好的块
    current_title:str |None              = None#正在处理的标题
    current_title_lines:list[str]    = []#当前标题下攒到哪些行了（临时篮子）
    is_code:bool                     = False#用于判断是否是在代码块里的##
    # 2.2 对清洗后的md_content进行按行切割 \n -> list[str]
    md_lines:list[str] = md_content.split("\n")
    # 2.3 定义正则匹配对应的标题
    """
    ^       从头匹配
    $       结尾匹配
    \s      空格
    *       量次 0 - n 
    +       量次 1 - n  有1-n的内容=有一些内容
    {1,2}   固定重复次数：1 - 2 次
    """
    title_re = re.compile(r"^\s*#{1,6}\s.+")
    # 2.4 循环进行每行处理
    for line in md_lines:

        # todo:  问题3: 空行处理一下! \n\n -> 跳过了
        line = line.strip()#空行先去空格
        if not line:
            logger.debug(f"当前行为空行,跳过,进行下一行处理!")
            continue
        logger.debug(f"当前为正常行,不跳过,正常处理!")

        # todo 问题4: md中可能有代码块 ## 注释  思路: 在代码块中的 正则匹配就是失效的! 出来就是可以的!
        if line.startswith("```") or line.startswith("~~~"):
            is_code = not is_code
            current_title_lines.append(line)
            continue

        # 2.5 查看是不是标题
        match = title_re.match(line)
        if not is_code and match:
            # 2.6 是-> 结算上一赛季 / 记录新的赛季
            # 结算上一赛季
            # todo: 问题1: 第一次碰到标题如何处理?  第一行就是标题, 不要结算!
            # todo: 问题2: 空标题?  一个标题但是没有内容!!  标题下没有内容! 不想要!
            if current_title and len(current_title_lines) > 1:
                chunks_title.append(
                    {
                        "title":current_title,
                        "content": "\n".join(current_title_lines)
                    }
                )

            # todo:  问题5: 孤儿数据的问题 - 没有标题的数据  文档使用非标题开始
            if not current_title and len(current_title_lines) > 0:
                current_title = line#line是正在处理的标题
                current_title_lines.append(line)
                continue

            # 记录新的赛季
            current_title = line
            current_title_lines = [line]#标题是当前段落第一节
        else:#普通行
            # 2.7 不是 -> 普通行 -> 追加到 current_title_lines 继续下一行处理即可
            current_title_lines.append(line)
    # 结算最后一个标题
    # 2.8 跳出循环,最后进行数据结算 [因为最后一个标题没有人触发结束]
    if current_title and len(current_title_lines) > 1:
        chunks_title.append(
            {
                "title": current_title,
                "content": "\n".join(current_title_lines)
            }
        )
    # 2.9 返回结果
    logger.info(f"将文档切成了{len(chunks_title)}块!")
    return chunks_title

"""
 improve_chunk_size(chunks:list[dict[str,Any]]) -> list[dict[str,Any]]:
        3.1 准备工作,定义接收的chunks,list[dict[str,Any]]
        3.2 chunks循环,获取每个chunk,content先判断长度是否超过600,超过,进行递归切割器切割 [_函数]
            条件: content > 600
        3.3 合并同一个标题下过短的chunk内容,content小于400 [_函数] 
            条件: 1. 同一个标题 2. content小于400 3. 不能大于1000
        3.4 返回处理长度后的chunks

"""
# 原则二和原则三: 合适大小和边界清晰(同一个标题进行切割，同一个标题进行合并)
def improve_chunk_size(chunks:list[dict[str,Any]]) -> list[dict[str,Any]]:
    logger.info(f"没有切割,也没有合并的原始数据长度:{len(chunks)}")
    # 3.1 准备工作,定义接收的chunks,list[dict[str,Any]]
    final_chunks:list[dict[str,Any]] = []
    # 3.2 chunks循环,获取每个chunk,content先判断长度是否超过600,超过,进行递归切割器切割 [_函数]
    for chunk in chunks:
        if len(chunk.get("content")) > 600:
            # 切割
            final_chunks.extend(_content_split(chunk))#extend可以把传入的 列表里的数据平铺进来不带[]
        else:
            # 直接使用
            final_chunks.append(chunk)
    logger.info(f"按标题切割,但是没有合并的数据长度:{len(final_chunks)}")
    # 3.3 合并同一个标题下过短的chunk内容,content小于400 [_函数]
    final_chunks = _content_merge(final_chunks)
    logger.info(f"按标题切割,也进行了合并的数据长度:{len(final_chunks)}")
    return final_chunks


def _content_split(chunk:dict[str,Any]) -> list[dict[str,Any]]:
    """
    本质就是用递归切割器进行切割!
       1. 数据前置处理
           chunk = {title:"标题",content:"标题\n行1 -> chunk  标题\n行2  -> chunk  标题\n行3 -> chunk"}
           content -> 标签\n -> 扣出去
           准备公共前缀 -> 标签\n
       2. 数据切割
           content
       3. 切割的数据组装成对应chunk
           chunk + 标签\n
    """
    current_title_chunks:list[dict[str,Any]] = []#定义的列表来存储chunks
    # 1. 数据前置处理
    content:str = chunk.get("content")
    title  :str = chunk.get("title")
    # 标题\n 标题len + 1
    content     = content[len(title)+1:]#content指的是把标题去掉的内容
    pre_fix:str = title+"\n"#标题

    # 2. 声明递归切割器
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。","！","？","；"],
        # config 四个常量
        chunk_size=CHUNK_SIZE - len(pre_fix),  # 600 - len(pre_fix)  这样算是因为最后每个chunk都要带标题
        chunk_overlap=CHUNK_OVERLAP
    )

    # 3. 递归切割器切割数据
    chunk_list:list[str] =  text_splitter.split_text(content)
    for index,chunk_content in  enumerate(chunk_list,start=1):
        # chunk_content -> chunk -> dict [str,Any]
        # chunk: parent_title -> 以前的title #警告 title -> 警告_1 , 警告_2  part 1 2
        #        content -> 内容 pre_fix + chunk_content
        sub_chunk = {
            "parent_title":title,
            "title":f"{title}_{index}", # 自己定义的规则 目标: 1. 让模型知道我们是同一个标题下的切块!还标上了序号 2. 后续我们自己展示的时候,是同一个标题的切块
            "part": index,
            "content": pre_fix+chunk_content
        }#这就是一个chunk
        current_title_chunks.append(sub_chunk)
    logger.info(f"已经完成{title}的内容切割!切割的数量:{len(current_title_chunks)}")
    return current_title_chunks


def _content_merge(final_chunks:list[dict[str,Any]]) -> list[dict[str,Any]]:
    """
      短了合并: 条件1: 同一个parent_title 条件2: content < 400  条件3: 合并后不能超过1000

      合并思路: list 最终切块结果  base_chunk 合并的基础   循环过程中 next_chunk 合并入的数据
               第一次循环 -> next_chunk -> base_chunk
               第二次循环 -> next_chunk  = 1. base_chunk < 400  2. base_chunk next_chunk  parent_title  3. 合并 < 1000  满足 next content ->  base content | 不满足  base -> list  next -> base
    """
    chunks:list[dict[str,Any]] = [] # 装最终合并的数据
    base_chunk:dict[str,Any]   = None # 当前还没有最终放入结果、但可能继续与下一个块合并的 chunk。
    logger.info(f"合并的数据长度:{len(final_chunks)}")
    # 进行数据循环处理
    for next_chunk in final_chunks:
        if not base_chunk:
            # 第一次循环,next->base
            base_chunk = next_chunk
            continue
        # 第二次循环开始 base有值  next是否能够合并即可
        # 条件1 base长度 < 400 条件2 同一个父标题  条件3 合并后的长度
        if len(base_chunk.get("content"))< CHUNK_MIN:
            # 小于400
            # 判断是不是同一个父标题! 天坑: 必须添加非空校验,否则会将两个独立的标题合并到一起!
            if base_chunk.get("parent_title")  and base_chunk.get("parent_title") == next_chunk.get("parent_title"):
                # 判断合并后的长度是否小于1000
                if (len(base_chunk.get("content")) + len(next_chunk.get("content"))) < CHUNK_MAX_SIZE:
                    # 真的可以合并
                    # part 1 4 6 7  -> 1 2 3 4
                    base_chunk['content'] = base_chunk.get("content") + "\n" + next_chunk.get("content")
                else:
                    chunks.append(base_chunk)
                    base_chunk = next_chunk
            else:
                chunks.append(base_chunk)
                base_chunk = next_chunk
        else:
            chunks.append(base_chunk)
            base_chunk = next_chunk

    # 跳出循环
    if base_chunk:
        # 最后一个一定没有添加进来!
        # 场景1: 前面满足 base-> chunks 最后一个变成base
        # 场景2: 前面不满足 base-> chunks 最后一个变成base
        chunks.append(base_chunk)
    logger.info(f"合并之后的数据长度:{len(chunks)}")
    return chunks

"""
append_chunk_meta_data(chunks:list[dict[str,Any]],file_title:str):
        4.1 循环chunks
        4.2 dict是否存在对应的key,不存在赋予默认值

"""
def append_chunk_meta_data(chunks:list[dict[str,Any]],file_title:str):
    # 4.1 循环chunks
    for chunk in chunks:
        chunk['file_title'] = file_title
        if "part" not in chunk:
            chunk['part'] = 1#当前 chunk 是原章节/标题下的第几段（第几个子块）。
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk['title']
    logger.info(f"完成所有chunk的元数据补充!")

"""
将chunks数据备份到本地
backup_chunks_json(chunks,md_path_obj:Path):
        5.1 chunks.json文件的Path对象 md_path_obj.parent / "chunks.json"
        5.2 对象.write_text(  json.dumps(chunks),"utf-8" )

"""
def backup_chunks_json(chunks,md_path_obj:Path):
    # 5.1 chunks.json文件的Path对象 md_path_obj.parent / "chunks.json"
    chunks_json_obj:Path = md_path_obj.parent / "chunks.json"
    # 5.2 对象.write_text(  json.dumps(chunks),"utf-8" )
    chunks_json_obj.write_text(data=json.dumps(chunks,indent=4,ensure_ascii=False),encoding="utf-8")
    """
       json 
          dump  python -> json   [备份]
             dumps  python -> json字符串 -> 我们 -> 后续在进行操作
             dump   python -> json字符串 ->直接 写到本地磁盘的位置 
          load  json   -> python [还原]
    """
    # 5.2 对象.write_text(  json.dumps(chunks),"utf-8" )
    # with chunks_json_obj.open(mode="w",encoding="utf-8") as f:
    #     json.dump(chunks,f,indent=4,ensure_ascii=False)
    logger.info(f"已经完成{md_path_obj.name}对应切换内容的数据备份!")




def split_document(state: ImportGraphState) -> ImportGraphState:
    # 1. 参数校验和数据获取
    md_content, file_title, md_path_obj = validates_and_get_data(state)
    # 2. 根据语义进行初步切割
    title_chunks:list[dict[str,Any]] = document_split_by_title(md_content)
    # 3. 进行同一个标题下的长度优化处理和添加重叠部分
    chunks:list[dict[str,Any]] = improve_chunk_size(title_chunks)
    # 4. 元数据的补充和完善
    append_chunk_meta_data(chunks,file_title)
    # 5. 数据备份
    backup_chunks_json(chunks,md_path_obj)
    # 6. 更新state
    state['chunks'] = chunks
    # 7. 返回结果
    return state