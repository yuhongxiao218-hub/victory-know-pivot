import time
from pathlib import Path

import requests

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS, MINERU_POLL_TIMEOUT_SECONDS, \
    MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger,PROJECT_ROOT
from app.infra.config.providers import infra_config
import shutil #shell utill Linux的工具

def vaildates_pdf_paths(state:ImportGraphState)->tuple[Path,Path]:
    """
    校验并规范化 PDF 解析所需的两个路径：pdf 文件路径 和 输出目录路径。

    1. 从 state 读取 pdf_path / local_dir 并做非空校验（空则抛 ValueError）
    2. local_dir 为空时兜底为 项目根/output
    3. 转成 Path 对象，校验 pdf_path 对应的文件存在（不存在抛 FileNotFoundError）
    4. 校验/创建输出目录（不存在则递归创建）

    :param state: 导入流程状态，需包含 pdf_path（待解析 PDF 地址）和 local_dir（输出目录地址）
    :return: (pdf_path_obj, local_dir_obj) —— PDF 文件 Path 对象 和 输出目录 Path 对象
    """
    #3.1.从state中获取pdf_path和local_dir地址
    pdf_path:str=state.get("pdf_path")
    local_dir:str=state.get("local_dir")
    #2.local_dir 为空时兜底为 项目根/output
    if not pdf_path:
        logger.error(f"pdf_path地址为空，业务无法继续，提前终止！")
        raise ValueError(f"pdf_path地址为空，业务无法继续，提前终止！")
    if not local_dir:
        local_dir=PROJECT_ROOT / "output"
        logger.warning(f"local_dir没有赋予默认值，配置默认输出地址为{local_dir},业务继续进行！")
        state['local_dir']=str(local_dir)  #state类型为字符串，得到的local_dir对象要转成字符串类型
    #3.
    pdf_path_obj:Path=Path(pdf_path)
    local_dir_obj:Path=Path(local_dir)
    #4.
    if not  pdf_path_obj.is_file():
        logger.error(f"虽然pdf的地址为{pdf_path}，但是检查没有真实对应到文件！业务无法继续，提前终止！")
        raise FileNotFoundError(f"虽然pdf的地址为{pdf_path}，但是检查没有真实对应到文件！业务无法继续，提前终止！")
    #5.
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir路径地址为{local_dir},但没有具体的文件夹，直接创建对应的文件夹，业务继续执行！")
        #parents可以创建多层文件夹,exist_ok如果存在也不报错，防御性的编程，预防多线程并发问题
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    return pdf_path_obj,local_dir_obj


def apply_mineru_get_uploader_url(pdf_file_name:str)->tuple[str,str]:#返回的是url和batch_id
    """
    向 MinerU 申请文件上传地址，返回 batch_id 和预签名上传 URL。

    :param pdf_file_name: 待解析 PDF 的完整文件名（包含扩展名）
    :return: (batch_id, upload_file_url) —— 解析任务编号 和 文件上传地址
    """
    #1.组装对应请求头，官网拿取
    token=infra_config.mineru_config.api_key
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    #2.组装请求体
    data = {
        "files": [
            {"name": pdf_file_name}
        ],
        "model_version": MINERU_MODEL_VERSION,
    }

    """
    补充官网的请求体参数seed和callback
    MinerU 文件解析流程：
    1. 向 MinerU 申请文件上传地址。
       MinerU 返回：
       - url：文件上传地址
       - batch_id：本次解析任务的编号
    2. 将具体文件上传到该地址：
       requests.put(url, data=文件数据)
    3. MinerU 会异步解析文件。
       需要根据 batch_id 查询当前解析状态。
    4. 解析完成后，MinerU 返回 zip_url，
       可通过 zip_url 下载解析结果。

    获取解析结果的两种方式：
    方案一：轮询
    提交任务 -> 获得 batch_id -> 定期查询解析状态
    -> 解析完成 -> 获得 zip_url -> 下载结果
    方案二：回调接口（Webhook）需要在请求体里写入callback:url
    提交任务时配置回调接口地址
    -> MinerU 异步解析
    -> 解析完成后，MinerU 主动调用回调接口
    -> 将解析结果传给我们
    注意：
    1. 回调接口地址必须是公网可以访问的地址。
    2. 回调是异步的：提交任务后不必一直等待，
       MinerU 解析完成时会主动通知我们的服务。
    """

    #3.向对应接口发送post方式请求，并等待[阻塞]获取结果【这个key是固定的有url/headers/params/json/data/timeout]
    url = f"{infra_config.mineru_config.base_url}/file-urls/batch"
    response=requests.post(
        url=url,
        headers=header,
        json=data,
        timeout= MINERU_DOWNLOAD_TIMEOUT_SECONDS
    )
    #4.判断http协议网络状态
    status_code=response.status_code
    if status_code!=200:
        logger.error(f"向minerU服务器申请解析地址，但是网络异常！状态码为：{status_code},业务无法继续，提前终止")
        raise RuntimeError(f"向minerU服务器申请解析地址，但是网络异常！状态码为：{status_code},业务无法继续，提前终止")
    #5.判断业务状态码
    #response->响应数据为json格式->拿到code
    #响应数据可能为json()/text/content
    response_json=response.json()
    code=response_json.get("code",-1)
    if code!=0:
        message=response_json.get("msg", "未知错误")
        logger.error(f"向minerU服务器申请解析地址失败，业务状态码：{code}，原因：{message}")
        raise RuntimeError(f"向minerU服务器申请解析地址失败，业务状态码：{code}，原因：{message}")

    #6.获取日志结果batch_id/upload_url地址
    json_data=response_json.get("data",{})
    batch_id:str=json_data.get("batch_id")
    upload_file_url:str=json_data.get("file_urls")[0]
    #7.返回结果
    return batch_id,upload_file_url

def upload_file_to_url(upload_url:str,pdf_path_obj:Path) -> None:
    """
    读取文件的操作，从pdf_path读取数据之后传到upload_url这个地址中
    :param upload_url:上传到的地址，上传文件用的预签名 URL（Presigned URL），可以不需要认证（因为授权信息已经由服务端提前签名并包含在 URL中），但仍然需要验证，校验通过，才能上传；否则通常返回 403 Forbidden。
    :param pdf_path_obj:pdf的地址
    :return: None（上传成功不返回内容，失败时抛 RuntimeError）
    """
    # 5.1 读取上传文件的数据pdf_path_obj->字节的数据
    data=pdf_path_obj.read_bytes()
    """
    Path：表示一个路径对象。

    常用路径属性：
    - .name：文件名（包含扩展名）
    - .stem：文件名主体（不包含扩展名）
    - .suffix：文件扩展名
    - .parent：上一级目录
    - .parents：所有上级目录

    常用判断方法：
    - .is_file()：判断路径是否为文件
    - .is_dir()：判断路径是否存在且是一个文件夹
    - .exists()：判断路径是否存在

    常用读写方法：
    - .read_text()：读取文本内容
    - .read_bytes()：读取字节内容
    - .write_text()：写入文本内容
    - .write_bytes()：写入字节内容

    常用遍历方法：
    - .iterdir()：遍历当前目录下的内容
    - .glob()：按指定规则查找当前目录中的文件(只能拿到一个)
    - .rglob("*.md")：按指定规则递归查找当前目录及所有子目录中的文件（能拿到所有）
    """
    # 5.2 向指定的地址发起put请求,传递pdf文件数据
    #put（地址，上传数据类型）这个数据类型有多种 json/data/params/headers,此次传数据选data
    #response=requests.put(url=upload_url,data=data,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    #with 是为了让 Session 用完后自动关闭和清理资源。
    with requests.Session() as session:
        session.trust_env=False
        response=session.put(url=upload_url,data=data,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    """
    requests 的普通请求方法，底层会临时创建并使用一个 Session；Session 则让你自己持有这个会话，以便重复使用。
    Session优点：1.复用请求 2.session.trust_env=False 让Session不要自动读取电脑系统环境变量中的网络配置，不增加别的请求头的信息。设置了什么就传什么，加大成功率
    """



    # 5.3 判断响应状态码,不是200抛出异常! [本次只需要判断http状态码,不需要判断业务code! 本身就是不是minerU业务接口,文件服务器的上传地址]
    status_code=response.status_code
    if status_code!=200:
        logger.error(f"向：{upload_url}地址上传文件，网络状态失败：{status_code},业务无法继续，提前终止！")
        raise RuntimeError(f"向：{upload_url}地址上传文件，网络状态失败：{status_code},业务无法继续，提前终止！")



def poll_mineru_zip_url(batch_id:str) -> str:
    """
    通过batch_id轮询获取解析结果,获取的结果是一个url -> zip文件下载的地址   pdf -> minerU解析 -> zip -> zip_url
    :param batch_id: MinerU 解析任务的编号，用于查询解析状态
    :return: 解析完成后 MinerU 返回的 zip 压缩包下载地址（full_zip_url）
    """
    #6.1 接收常量参数 临时变量接收值 最大轮询时间 / 轮询间隔时间 / 记录起始时间
    max_timeout=MINERU_POLL_TIMEOUT_SECONDS#最大轮询时间
    interval_time=MINERU_POLL_INTERVAL_SECONDS
    start_time=time.time()#取出当前时间 单位为秒

    url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {infra_config.mineru_config.api_key}"
    }

    #6.2 死循环 while True:
    while True:
       # 6.2.1 有没有超过最大等待时间 [1~N]
       if time.time()-start_time>=max_timeout:
           logger.error(f"轮询获取解析结果时间超时！等待时间为：{time.time()-start_time},最大等待时间：{max_timeout},业务无法继续，提前终止")
           raise TimeoutError(f"轮询获取解析结果时间超时！等待时间为：{time.time()-start_time},最大等待时间：{max_timeout},业务无法继续，提前终止")
       #6.2.2 向指定的地址请求查询解析状态。网络/代理的瞬时异常不应直接中断
       #整个 MinerU 任务；参考项目在这里会等待后继续轮询。
       try:
           response=requests.get(url,headers=header,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
       except requests.RequestException as exc:
           logger.warning(f"轮询获取解析结果时网络异常：{exc}，稍后重试！")
           time.sleep(interval_time)
           continue
       #6.2.3 判断网络状态码status_code != 200  [判断是否可以给与机会(5xx),可以等待重试 / 其他的也需要抛出异常终止轮询 ]
       status_code=response.status_code
       if status_code != 200:
           # 错误可以重试 5xx 我们是客户端请求服务端报5xx
           if 500<=status_code<600:
               logger.warning(f"轮询获取解析结果，出现服务器异常，状态码为：{status_code},稍后重试！")
               time.sleep(interval_time)#线程休眠！因为
               continue
           #错误不可以重试
           logger.error(f"轮询获取解析结果，出现异常，状态码为：{status_code},异常是不可修复和重试，业务无法继续，提前终止")
           raise RuntimeError(f"轮询获取解析结果，出现异常，状态码为：{status_code},异常是不可修复和重试，业务无法继续，提前终止")
       #  6.2.4 判断业务状态码code != 0 [需要抛出异常终止轮询]
       json_data=response.json()
       code=json_data.get("code",-1)
       if code!=0:
           logger.error(f"轮询获取解析结果，业务状态出现异常，code为：{code},异常是不可修复和重试，业务无法继续，提前终止")
           raise RuntimeError(f"轮询获取解析结果，业务状态出现异常，code为：{code},异常是不可修复和重试，业务无法继续，提前终止")
       #  6.2.5 判断文件的解析状态state
       #           done -> zip -> return
       #           fail -> 失败 -> raise
       #           else -> 休眠 再次请求
       extract_results = json_data.get("data", {}).get("extract_result", [])
       if not extract_results:
           logger.error(f"MinerU 未返回解析结果，batch_id={batch_id}，响应数据：{json_data}")
           raise RuntimeError(f"MinerU 未返回解析结果，batch_id={batch_id}")

       result=extract_results[0]
       state=result.get("state")
       if state=="done":
           #解析完成了
           full_zip_url:str=result.get("full_zip_url")
           logger.info(f"文件解析成功，完成解析，下载地址为{full_zip_url}")
           return full_zip_url
       elif state=="failed":
           #解析失败了
           error_message = result.get("err_msg", "MinerU 未返回具体失败原因")
           logger.error(
               f"MinerU 文件解析失败，batch_id={batch_id}，"
               f"file_name={result.get('file_name', '')}，原因：{error_message}"
           )
           raise RuntimeError(
               f"MinerU 文件解析失败，batch_id={batch_id}，原因：{error_message}"
           )
       else:
           #正在解析
           logger.info(f"文件正在解析中，当前状态为：{state},稍后重试！")
           time.sleep(interval_time)
           continue



#stem是pdf不带后缀名的名称上
def download_and_extract_md(zip_file_url:str,local_dir_obj:Path,stem:str) -> str:
    """
    下载 MinerU 解析结果的 zip 压缩包，解压到输出目录，并返回解压后的 Markdown 文件路径。

    :param zip_file_url: zip 压缩包的下载地址（MinerU 解析完成后返回的 full_zip_url）
    :param local_dir_obj: 输出文件夹地址（Path 对象）
    :param stem: 看是不是等于原文件名，希望解压后的文件名的于原文件名，state也就是解压后得到的文件名，可能mineru解压后是full.md也要进行校验，将他转成真实的原文件的文件名
    :return: 解压后 Markdown 文件的地址（str）
    """
    #7.1 向指定的zip下载地址发起请求(get) 响应字节数据(运行内存)
    response=requests.get(zip_file_url,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    status_code=response.status_code
    if status_code!=200:
        logger.error(f"从{zip_file_url}下载文件出现网络异常，状态码为{status_code}业务无法继续，提前终止！！")
        raise RuntimeError(f"从{zip_file_url}下载文件出现网络异常，状态码为{status_code}业务无法继续，提前终止！！")
    #7.2 创建一个存储zip数据的文件 output / stem_result.zip
    zip_file_obj:Path=local_dir_obj /f"{stem}_result.zip"
    #7.3 将字节数据写到zip的文件中 (write_bytes(data))
    data=response.content   # 字节内容content（bytes）→ 下载 zip 这种二进制文件用这个  #文本内容text（str，按编码解码）
    zip_file_obj.write_bytes(data=data)#往zip_file_obj里写完内容后才算真的创建完成
    #7.4判断解压缩的文件地址是否存在 output / stem
    zip_file_dir:Path=local_dir_obj / stem
    #7.5存在先清空, 删除
    if zip_file_dir.is_dir():
        # 将文件本身和内容全部清空
        """
        rmtree
        """
        shutil.rmtree(zip_file_dir)#递归删除 删除自己内部的文件，删除自己本身
    #7.6 真实创建对应的文件夹  output / stem
    zip_file_dir.mkdir(parents=True,exist_ok=True)
    # 7.7 zip文件解压到对应输出文件夹
    """
    unpack_archive
    #参数1：要解压的文件Path对象或者字符串对象
    #参数2：要解压到那个文件夹地址 不要制定解压类型会自动识别
    """
    shutil.unpack_archive(zip_file_obj,zip_file_dir)
    """
    shutil.make_archive做压缩的
    参数1：压缩后的文件名info
    参数2:压缩后的格式zip
    参数3:压缩哪个文件夹
    """
    #7.8 检查输出文件夹中是否存在md文件 没有 -> 终止业务
    # 思路 1：获取文件夹中的所有文件和子文件夹，再比较后缀名。
    # List[Path] = Path.iterdir()  # 获取文件夹下的所有文件或文件夹
    # 思路 2：获取文件夹下指定类型的文件。
    # Path.glob("*.md")  # 例如：full.md、xx.md
    """
    rglob返回的是yield类型
    yield：函数不会一次结束，而是变成“生成器”，每次迭代时产出一个值，并记住当前位置，等待下一次继续。
    """
    md_list:list[Path]=list(zip_file_dir.rglob("*.md"))
    if not md_list:
       logger.error(f"已将文件解压到{local_dir_obj}文件夹，但是内部没有md文件，业务无法继续进行，提前终止！！")
       raise RuntimeError(f"已将文件解压到{local_dir_obj}文件夹，但是内部没有md文件，业务无法继续进行，提前终止！！")
    #以下就是进行命名规整，统一叫原始文档名.md
    # 7.9 检查md文件是否叫 stem.md , 是直接返回即可 Path -> str
    for md_file in md_list:
        if md_file.stem==stem:
            logger.info(f"解压后直接存在原文件名{md_file.name}对应的md文件，无需重命名，直接返回即可")
            return str(md_file)
    # 7.10 再检查叫不叫full.md(这个是mineru官方命名的) -> 重命名 -> 新地址返回
    target_md_obj:Path|None=None
    for md_file in md_list:
        if md_file.stem=="full":
            target_md_obj=md_file
            break
    #重写命名规则
    if  target_md_obj:
         # 找到了full.md,写重命名规则
         # 将full.md->stem.md  使用rename 注意：返回值才是修改后的地址
         #问题：rename后的文件会放在当前py目录下的地址，因为使用了相对地址，
         #解决：使用绝对地址 Path .with_name()      绝对地址.../full.md->绝对地址.../xx.md
         #新的目标绝对地址=要修改的绝对地址对象.with_name(要修改的目标文件名)
         new_target_md_obj:Path=target_md_obj.with_name(f"{stem}.md")
         new_md_path:Path=target_md_obj.rename(new_target_md_obj)
         logger.info(f"已经完成full.md重命名，新的文件地址为：{new_md_path}")
         return  str(new_md_path)
    else:
        logger.error(f"解压后的文件夹中存在{len(md_list)}个md文件，但是没有叫原文件名{stem}/full.md文件，也无法确定哪个文件，业务提前终止")
        raise RuntimeError(f"解压后的文件夹中存在{len(md_list)}个md文件，但是没有叫原文件名{stem}/full.md文件，也无法确定哪个文件，业务提前终止")


def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir

    :param state: 导入流程状态，需包含 pdf_path（待解析 PDF 地址）和 local_dir（输出目录地址）
    :return: 更新后的 state（回写 md_path）
    """
    #1.进行参数合理性调验
    pdf_path_obj,local_dir_obj=vaildates_pdf_paths(state)
    #2.向minerU申请文件上传地址和batch_id
    # MinerU 依赖文件扩展名识别格式，必须传完整文件名。
    batch_id,upload_file_url=apply_mineru_get_uploader_url(pdf_path_obj.name)
    logger.info(
        "MinerU submitted: batch_id={}, file={}, size={} bytes",
        batch_id, pdf_path_obj.name, pdf_path_obj.stat().st_size,
    )
    #3.向指定文件地址(预签名地址)上传文件数据
    upload_file_to_url(upload_file_url,pdf_path_obj)
    #4.轮询获取解析结果
    full_zip_url=poll_mineru_zip_url((batch_id))
    #5.下载并解压和重命名文件
    md_path:str=download_and_extract_md(full_zip_url,local_dir_obj,pdf_path_obj.stem)
    #6.更新state的md_path
    state['md_path']=md_path
    return state
