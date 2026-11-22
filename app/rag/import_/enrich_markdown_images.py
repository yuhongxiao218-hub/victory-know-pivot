import base64
import logging
import re
from mimetypes import guess_type
from pathlib import Path

from dill import objects
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio import Minio
from minio.deleteobjects import DeleteObject

from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import IMAGE_CONTEXT_CHAR_LENGTH, SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.llm.providers import  llm_provider



def validates_and_get_data(state:ImportGraphState) -> tuple[str,Path,Path]:
    """
    校验并读取 Markdown 文件内容，返回图片处理所需的三个数据。

    :param state: 导入流程状态，需包含 md_path（Markdown 文件地址）
    :return: (md_content, md_path_obj, images_path_obj) —— md 正文内容、md 文件 Path 对象、images 目录 Path 对象
    """
    #1.1 获取md_path参数从state
    md_path:str=state.get("md_path")
    #1.2 md_path的非空校验
    if not md_path:
        logger.error(f"md_path内容为空，业务无法继续，提前终止")
        raise ValueError(f"md_path内容为空，业务无法继续，提前终止")
    #1.3 转成md_path_obj Path对象,做存在性校验! is_file()
    md_path_obj:Path=Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path地址有值：{md_path},但是没有对应的文件，业务无法继续，提前终止！")
        raise FileNotFoundError(f"md_path地址有值：{md_path},但是没有对应的文件，业务无法继续，提前终止！")
    #1.4 使用md_path对象读取md_content,更新state
    md_content:str=md_path_obj.read_text(encoding="utf-8")
    state["md_content"]=md_content
    #1.5 md_path_obj可以获取images path地址
    images_path_obj:Path=md_path_obj.parent /"images"
    #1.6 顺序返回 return md_content,..
    return md_content,md_path_obj,images_path_obj

def scan_images_info(images_path_obj: Path, md_content: str) -> list[tuple[str, Path, tuple[str, str]]]:
    """
    扫描 images 目录下的所有图片，匹配其在 md 正文中的引用，并提取每张图片的上下文。

    :param images_path_obj: images 图片目录的 Path 对象
    :param md_content: Markdown 正文内容（用于正则匹配图片引用）
    :return: 图片信息列表，元素为 (图片文件名, 图片 Path 对象, (上文, 下文))
    """
    #2.1 获取images下的所有图片Path对象 [列表/生成器]
    #iterdir() |glob()|rglob()
    generator=images_path_obj.iterdir()
    images_info_list:list[tuple[str, Path, tuple[str, str]]]=[]
    #2.2 循环每一张图片对象
    for image_file_obj in generator:
        #是不是图片的检查
        if image_file_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.debug(f"{image_file_obj.name}不是图片，跳过，进行下一张的处理！")
            continue
        #2.2.1 定义这个图片的正则规则
        image_name:str=image_file_obj.name
        #参数1：定义的正则，每次都不同，没有必要提前编译
        #参数2：正则生效的字符串
        # 2.2.2 使用正则到md_content中进行匹配
        match=re.search(r"\!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)",md_content)
        #2.2.3 检查是否匹配 Match是否为空
        if not match:
            logger.debug(f"{image_name}没有在md_content被使用，跳过，进行下一张处理！")
            continue
        #2.2.4 获取对应上下信息
        start=match.start()
        end=match.end()
        #max是指不能取负值，min是指不能超过最大长度
        pre_context:str=md_content[max(start-IMAGE_CONTEXT_CHAR_LENGTH,0):start]
        post_context:str=md_content[end+1:min(end+IMAGE_CONTEXT_CHAR_LENGTH,len(md_content))]
        #2.2.5 组装当前的数据元组 (图片名字str,图片Path对象,(上文,下文)) 添加到大的列表
        images_info_list.append(
            (
                image_name,
                image_file_obj,
                  (
                    pre_context,
                    post_context
                  )
            )
        )
    logger.info(f"已经完成了{len(images_info_list)}图片内容的匹配")
    #2.3 返回列表数据
    return images_info_list



def summary_image_info(image_info:list[tuple[str,Path,tuple[str,str]]],root_folder:str) -> dict[str,str]:
    """
    :param image_info: 图片的信息 scan_images_info获得
    :param root_folder: 这个是要解析的文件的目录名 ej:hak180说明书
    :return: 图片名到图片描述的映射字典 dict[str, str]（key 为图片文件名，value 为视觉模型生成的摘要）
    """
    #3.1 数据准备工作 初始化一个视觉模型对象 | 定义一个接收数据的字典
    image_summary:dict[str,str]={}
    version_model=llm_provider.vision_chat()
    #3.2 循环处理每一张图片    for str,Path,context in  list[tuple[str,Path,tuple[str,str]]]
    for image_name, image_path_obj, image_context in image_info:
        #  3.3 封装模型提示词信息
        #3.3.1 图片的描述 包装上下文信息提示词
        image_des:str= load_prompt("image_summary",image_content=image_context,root_folder=root_folder)
        image_type:str=guess_type(image_name)[0]
        image_base64:str=base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": image_des},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_type};base64,{image_base64}"},
                },
        ]
        )
        #3.4 封装调用链: 直接 model.invoke | 调用链 提示词 | 模型 | 解析器 OutParser()
        chains=version_model | StrOutputParser()
        #3.5 使用调用链进行调用和获取结果    必须传进去的是一个消息列表
        try:
            summary:str=chains.invoke([message])#这是图片的信息
        except Exception as exc:
            logger.warning(f"图片{image_name}描述生成失败，跳过该图片描述：{exc}")
            continue
        """
         # chains = vision_model | StrOutputParser()
        # model.invoke([]) -> response -> response.content -> 字符串
        #使用chain可以直接拿到字符串
        # chains = vision_model | JSONOutputParser()
        # model.invoke([]) -> response -> response.content -> 字符串 -> json.loads(字符串) -> dict !需要将字符串转化为字典
        """
        #3.6 添加到字典中 {image_name,image_summary}
        image_summary[image_name]=summary#字典赋值
        logger.debug(f"图片{image_name},已经通过视觉模型进行意图识别，结果位{summary}")
    logger.info(f"已经完成{root_folder}文件夹里所有图片内容的总结，总结内容的长度为：{len(image_summary)}")
    return image_summary

def upload_image_to_minio(image_info:list[tuple[str,Path,tuple[str,str]]],stem) -> dict[str,str]:#stem是文件夹的名字
    #4.1 前置准备工作 获取minio_client / 创建一个存储结果的dict
    #todo:用来收集「图片名 → MinIO 访问地址」的映射：
    images_url:dict[str,str]={}
    minio_client:Minio=minio_gateway.client()
    #4.2 根据桶信息查询对应旧图片对象
    objects:list=minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        # todo 注意：桶名->/图片前缀/文件名/<-图片.png  决定「删谁」（桶内目录前缀）
        prefix=minio_gateway.image_dir[1:]+"/"+stem,#桶名->/图片前缀/文件名/<-图片.png
        recursive=True)
    #4.3 批量删除旧图片对象
    if objects:
        #查询到了有之前相应的图片进行删除
        """
        1. MinIO 对象类型
            Object：查询到的对象。
            DeleteObject：用于删除的对象。

        2. 删除对象前需要转换类型
            list_objects() 返回的是 Object 对象列表。
            remove_objects() 需要 DeleteObject 对象列表。

            转换过程：
                list[Object] -> object.name -> list[DeleteObject]

        3. remove_objects() 的注意事项
            remove_objects() 返回生成器。
            需要循环处理返回结果，删除操作才会真正触发。

            errors = remove_objects(...)
            for error in errors:
                pass
        """
        delete_object_list=[DeleteObject(obj.object_name) for obj in objects]#obj.object_name 是关键：它才是桶里那张图的完整 key，没有它，SDK 不知道删什么
        errors=minio_client.remove_objects(#remove_objects 返回一个「错误迭代器」：删除失败的对象 → 产出一个 DeleteError（里面有 object_name、错误码、错误信息）
            bucket_name=minio_gateway.bucket_name,
            delete_object_list=delete_object_list
        )
        for error in errors:
            logger.warning(f"删除{stem}文件夹下的图片出现问题，问题为：{error}")
    # 4.4 循环处理每一张图片
    for image_name, image_path_obj, _ in image_info:
        # 4.5 图片分别先实现，然后按照图片的名字，下一个来！
        try:
            # 4.6 调用 minio_client.fput 方法将图片上传到 MinIO 服务器
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=minio_gateway.image_dir + "/" + stem + "/" + image_name,
                file_path=str(image_path_obj),
                content_type=guess_type(image_name)[0],
            )

            # 不报错，成功
            # 4.7 根据自己的图片 infra 构建图片的访问地址
            url: str = minio_gateway.build_image_url(stem, image_name)
            #4.8 将图片名和访问地址添加到字典中
            images_url[image_name] = url
            logger.debug(f"图片{image_name}已经上传到MinIO，访问地址为：{url}")
        except Exception as e:
            logging.exception(f"{image_name}图片上传失败，跳过本次，进行下一张图片处理！")
            continue
    # 4.9 返回字典数据
    logger.info(f"已经完成{stem}文件夹下所有图片的上传，上传成功的图片数量为：{len(images_url)}")
    return images_url
"""
     目标: md_content内容进行替换 返回替换后的md_content ![](./) -> ![网络地址](http...)
        replace_old_md_content(md_content:str,images_url:dict[str,str],images_summary:dict[str,str]) -> str:
        5.1 循环images_url可以获取每张图片的image_name / 网络地址
        5.2 可以根据image_name去images_summary获取对应的summary描述
        5.3 定义提前编译查询原有图片标签的正则 re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)"
        5.4 使用正则的替换函数即可    替换后的内容 md_content  =  编译的对象.sub("替换入的内容" f"![summary](url)" ,"什么中查找替换" -> md_content)
        5.5 返回替换后的md_content
     6. 将新md_content写到本地磁盘,持久化
        目标: 文件名/文件名.md [old] 新的md_content->写到本地磁盘!!  文件名_new.md,最终返回一个新的md_path:str
        backup_md_return_path(md_content:str,md_path_obj:Path)
        6.1 获取新备份位置的Path 文件名_new.md  md_path_obj.with_name | rename
        6.2 向新的path对象写入新的md_content md_path_obj.write_text
        6.3 新的path对象转成字符串返回即可
     7. 更新state信息md_path -> 新的地址  md_content -> 新的内容
        state['md_path'] = 新地址
        state['md_content'] = 新content内容
     8. 返回state
        return state
"""
#字典想拿到key:value需要.items()方法

def replace_old_md_content(md_content: str,images_url: dict[str, str],images_summary: dict[str, str],) -> str:
    """
     目标: md_content内容进行替换 返回替换后的md_content ![](./) -> ![网络地址](http...)
    :param md_content:
    :param images_url:
    :param images_summary:
    :return:
    """
    #5.1 循环images_url可以获取每张图片的image_name / 网络地址
    for image_name, url in images_url.items():
        #5.2 可以根据image_name去images_summary获取对应的summary描述
        summary = images_summary.get(image_name, "图片")
        #5.3 定义提前编译查询原有图片标签的正则 re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)"
        # 匹配 Markdown 图片/链接中包含该文件名的内容
        # re.escape(image_name) 确保文件名按普通字符串精确匹配
        image_re = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)"
        )
        #5.4 使用正则的替换函数即可    替换后的内容 md_content  =  编译的对象.sub("替换入的内容" f"![summary](url)" ,"什么中查找替换" -> md_content)
        # 将旧链接替换成：![图片摘要](新图片URL)
        #sub替换方式：正则.sub(替换成什么, 在哪里替换)

        # 天然一个问题：sub(" \1 -> 反 / -> 正斜杠  替换中代表分组!!匹配错误!")
        # 解决分组替换：lambda 表达式，匿名函数
        # sub(1（替换入的字符串 -> 会检查 / 分组替换的特殊符号 | 匿名函数，匿名函数的返回结果替换入 -> 不会检查返回结果）, )
        md_content :str = image_re.sub(
            lambda _: f"![{summary}]({url})",
            md_content,#循环每处理一张图，就更新一次 md_content；下一轮会基于上一轮已经替换好的文本继续处理。
        )
    return md_content

def backup_md_return_path(md_content:str,md_path_obj:Path)->str:
    """
     目标: 文件名/文件名.md [old]变成新的md_content->写到本地磁盘!!  文件名_new.md,最终返回一个新的md_path:str
    :param md_content:
    :param md_path_obj:
    :return:
    """
    #6.1 获取新备份位置的Path 文件名_new.md  md_path_obj.with_name | rename会真的把磁盘上的文件重命名或移动。
    new_md_path_obj:Path=md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    #with_name() 是 pathlib.Path 的：保持原来的文件夹路径和扩展名，只替换文件名主体。
    #6.2 向新的path对象写入新的md_content md_path_obj.write_text
    new_md_path_obj.write_text(data=md_content,encoding="utf-8")
    #6.3 新的path对象转成字符串返回即可
    return str(new_md_path_obj)

def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content

    :param state: 导入流程状态，需包含 md_path（Markdown 文件地址）
    :return: 更新后的 state（回写 md_content，图片地址替换为 MinIO 地址并补充说明）
    """
    #1.获取参数并校验
    md_content,md_path_obj,images_path_obj=validates_and_get_data(state)
    #md没有图片，要做特殊校验和提前终止 mages_path_obj是一个文件夹 |iterdir() 遍历所有的文件或者子文件夹|glob rglob(*.后缀名) ->返回生成器
    if not list(images_path_obj.iterdir()):
        logger.info(f"{md_path_obj}中没有引用任何图片，无需后续图片处理，直接跳转到下一个节点！")
        return state
    #2.调用参数进行图片信息识别
    images_info_list:list[tuple[str,Path,tuple[str,str]]]=scan_images_info(images_path_obj,md_content)
    #3.调用模型进行图片内容总结
    image_summary:str=summary_image_info(images_info_list,md_path_obj.stem)
    #调用图片上传，获取图片上传地址
    images_url:dict[str,str]=upload_image_to_minio(images_info_list,md_path_obj.stem)
    #5.进行md_content内容进行替换 返回替换后的md_content ![](./) -> ![网络地址](http...)
    new_md_content:str=replace_old_md_content(md_content,images_url,image_summary)
    #6.将新md_content写到本地磁盘,持久化
    new_md_path:str=backup_md_return_path(new_md_content,md_path_obj)
    #7.更新state信息md_path -> 新的地址  md_content -> 新的内容
    state['md_path']=new_md_path
    state['md_content']=new_md_content

    return state
