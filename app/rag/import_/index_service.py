from app.process.import_.agent.state import ImportGraphState

from typing import Any

from pymilvus import MilvusClient, DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

"""
   步骤1: 参数校验和获取
     validate_and_get_chunks(state:ImportGraphState)->list[dict[str,Any]]:
     1.1 在state中获取embeddings_content数据
     1.2 进行embeddings_content非空校验
     1.3 返回embeddings_content即可
     步骤2: 前置检查集合是否存在,不存在创建即可即可
     prepared_item_name_collection():
     2.1 检查集合是否存在,存在不需要创建直接返回
     2.2 创建集合对应schema
     2.3 创建集合对应index
     2.4 创建集合
     2.5 加载集合
     步骤3: 数据更新和插入(删除旧的数据/插入新的数据)
     3.1 检查是否存在对应旧的数据,存在就删除(file_title)
     3.2 重新批量插入新的数据
     3.3 打印日志,插入是否成功即可
     步骤4: 返回state

"""

# 步骤1: 参数校验和获取
@step_log("validate_and_get_chunks")
def validate_and_get_chunks(state:ImportGraphState)->list[dict[str,Any]]:
    # 1.1 在state中获取embeddings_content数据
    embeddings_content:list[dict[str,Any]] = state.get("embeddings_content")
    # 1.2 进行embeddings_content非空校验
    if not embeddings_content:
        logger.error(f"embeddings_content内容为空,业务无法继续,提前终止!!")
        raise ValueError(f"embeddings_content内容为空,业务无法继续,提前终止!!")
    # 1.3 返回embeddings_content即可
    return embeddings_content

#  步骤2: 提前准备集合
@step_log("prepared_item_name_collection")
def  prepared_item_name_collection():
    # 4.1 检查集合是否存在,存在不需要创建直接返回
    milvus_client:MilvusClient =  milvus_gateway.client
    has:bool =  milvus_client.has_collection(collection_name=milvus_gateway.chunks_collection)
    if has:
        logger.info(f"{milvus_gateway.chunks_collection}已经存在,无需创建,直接返回!")
        return
    # 4.2 创建集合对应schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )

    # schema添加字段  item_name -> pk file_title item_name dense_vector sparse_vector
    # datatype= pymilvus_DataType.类型
    # schema.add_field(field_name="age", datatype=DataType.INT64, nullable=True, default_value=18)
    schema.add_field(field_name="chunk_id",datatype=DataType.INT64,is_primary=True)
    schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=600)
    schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=600)
    schema.add_field(field_name="parent_title",datatype=DataType.VARCHAR,max_length=600)
    schema.add_field(field_name="title",datatype=DataType.VARCHAR,max_length=600)
    schema.add_field(field_name="part",datatype=DataType.INT8)
    # max_length=2400 单位是字节  -utf-8mb4-   content 600
    schema.add_field(field_name="content",datatype=DataType.VARCHAR,max_length=65535)
    # 稠密向量 -> [] -> 元素类型 -> FP32  | FP16 (cpu/gpu)  || 使用dim 参数指定密集向量的维度。 (嵌入式模型)
    schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
    schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 4.3 创建集合对应index
    index_params = milvus_client.prepare_index_params()

    # 向量场
    # 稠密
    index_params.add_index(
        field_name="dense_vector",  # 给哪个字段的字段名字
        index_type="HNSW",          # 索引类型  FLAT 蛮力全局检索,召回率100% 慢 | 准
                                    #         IVF_FLAT  将数据分类,每类有一个中心点,先找中心点所在类,类中进行检索 快 / 精度少差
                                    #         HNSW  逐层向下精准定位 效率高于FLAT 精度高于IVF_FLAT
        index_name="dense_vector_index",  # 索引名字 [可选]
        metric_type="COSINE",             # 向量场相似度计算方式  稠密: [cosine] ip l2 稀疏: [ip] / bm25
        index_params={
            # 索引的额外参数
            "M": 64,  #  每个点最大连接的邻居的数量
            "efConstruction": 100  # 从多少个点中选出邻居的数量
        }
    )

    #稀疏
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # 稀疏倒排索引  从 数据->向量 倒过来 向量坐标关联数据 每次查询时候,先定位坐标关联的数据集合! 在进行集合内检索!
        index_name="sparse_vector_index",
        metric_type="IP",
        index_params={
            "inverted_index_algo": "SINDI"  # IP提高检索效率的方式
        }

    )
    # 4.4 创建集合
    milvus_client.create_collection(
        collection_name=milvus_gateway.chunks_collection,
        schema=schema,
        index_params=index_params
    )


    # 4.5 加载集合 从磁盘 加载到内存
    milvus_client.load_collection(
        collection_name=milvus_gateway.chunks_collection
    )

# insert_or_update_embeddings_content(embeddings_content:list[dict[str,Any]])
#      3.1 检查是否存在对应旧的数据,存在就删除(file_title)
#      3.2 重新批量插入新的数据
#      3.3 打印日志,插入是否成功即可
@step_log("insert_or_update_embeddings_content")
def insert_or_update_embeddings_content(embeddings_content:list[dict[str,Any]]):
    # 3.1 检查是否存在对应旧的数据,存在就删除(file_title)
    file_title:str = embeddings_content[0].get("file_title")
    milvus_client:MilvusClient = milvus_gateway.client
    delete_result = milvus_client.delete(
        collection_name=milvus_gateway.chunks_collection,
        filter=f"file_title == '{file_title}'"
    )
    delete_count:int=delete_result.get("delete_count",0)
    logger.info(f"删除{file_title}下对应的{delete_count}条数据!")
    # 3.2 重新批量插入新的数据
    insert_result =  milvus_client.insert(
        collection_name=milvus_gateway.chunks_collection,
        data=embeddings_content
    )
    insert_count:int = insert_result.get("insert_count",0)
    logger.info(f"已经完成数据插入,本次插入:{insert_count}条数据!")
    # 主键回显: embeddings_content -> chunk_id -> 主键 -> 主键回显
    # "ids": res.primary_keys,
    ids:list = insert_result.get("ids",[])
    # zip([1,2],[a,b,c,d])  1 a  2 b  zip以短为准! 如何做到以长为准呢? [中午思考下!!]
    for id,chunk in zip(ids,embeddings_content):
        chunk['chunk_id'] = id

@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果

    :param state: 导入流程状态，需包含 item_name 和带向量的 chunks
    :return: 更新后的 state（回写 chunk_id 等入库结果）
    """
    # 1. 获取并校验数据
    embeddings_content:list[dict[str,Any]] = validate_and_get_chunks(state)
    # 2. 提前准备chunk存储的集合
    prepared_item_name_collection()
    # 3. 进行数据更新或者插入
    insert_or_update_embeddings_content(embeddings_content)
    # 4. 修改和返回结果
    return state
