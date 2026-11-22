import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pymilvus import AnnSearchRequest

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import ITEM_NAME_CONFIRM_THRESHOLD, ITEM_NAME_CANDIDATE_THRESHOLD, ITEM_NAME_OPTIONS_TOPK
from app.shared.runtime.logger import logger
from app.shared.runtime.load_prompt import load_prompt
from app.infra.persistence.history_repository import history_repository
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.shared.model.lm_utils import require_json_object


# 显式型号通常由大写字母和至少两位数字组成，例如 LA2608、HAK180。
# 保留用户原文中的型号，避免 LLM 将“拨号访问组”等配置对象误当成知识库主体。
_EXPLICIT_MODEL_PATTERN = re.compile(
    r"\b(?=[A-Z0-9-]{4,}\b)(?=[A-Z0-9-]*[A-Z])(?=(?:[A-Z0-9-]*\d){2})[A-Z0-9-]+\b"
)

# 只补充检索术语，不替换用户原问题，防止查询归一化改变原始语义。
_QUERY_TERM_ALIASES = {
    "无线网络名称": "SSID",
}


def preserve_explicit_model_identifiers(item_names: list[str], original_query: str) -> list[str]:
    """将原问题里的显式设备型号合并回 LLM 提取结果，并保持顺序去重。"""
    merged: list[str] = []
    seen: set[str] = set()
    for item_name in [*item_names, *_EXPLICIT_MODEL_PATTERN.findall(original_query)]:
        normalized = item_name.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            merged.append(normalized)
            seen.add(key)
    return merged


def enrich_rewritten_query(original_query: str, rewritten_query: str) -> str:
    """将领域同义表达归一化为知识库使用的规范检索词。"""
    enriched = rewritten_query or original_query
    for alias, canonical in _QUERY_TERM_ALIASES.items():
        if alias not in original_query and alias not in enriched:
            continue
        if alias in enriched:
            enriched = enriched.replace(alias, canonical)
        elif canonical.casefold() not in enriched.casefold():
            # LLM 改写时可能丢掉原始同义词；此时补回规范术语。
            enriched = f"{enriched} {canonical}"
    return enriched


"""
1. 获取数据和参数校验(session_id/original_query)
  validate_and_get_data(state:QueryGraphState) -> tuple[str,str]:
  1.1 state获取数据
  1.2 非空校验  空 -> ValueError / 非空
  1.3 非空顺序返回值即可
2. 获取[有效的]聊天记录信息(session_id)
  valid_chat_message_list(session_id:str) -> list[dict]:
  2.1 调infra查询近期聊天数量 = 20  list[dict] -> {_id,role = "user / assistant",text,rewritten_query,item_names,image_urls,ts}
  2.2 过滤有效的聊天记录 条件1: 必须有item_names | 条件2: 过滤 role = user
  2.3 返回有效的聊天记录
3. 拼接提示词调用模型一次性识别item_names(不是最终)和重写问题(返回结构化数据json)
  call_llm_confirm_item_name_and_rewritten_query(message_list:list[dict],original_query:str) -> dict[str,str]
  3.1 infra获取模型的客户端对象(json_mode)
  3.2 准备提示词需要的前置参数(query / history_text  "没有有效聊天记录!!" <-  <- message_list  index,原始问题: text,重写后的问题: rewritten_query, 关联的item_name:[1,2,3]\n )
  3.3 加载提示词 shared/load...
  3.4 创建userMessage [ HumanMessage(content="xx") ]
  3.5 创建调用链 chains = llm | JSONOutputParser()
  3.6 执行调用链 dict =  chains.invoke(xx)
  3.7 数据检验 if "item_names" not int dict: .... []
  3.8 返回结果dict
4. 使用item_names进行向量数据库数据查询,获得每个item_name(模型)真实关联向量数据库的item_name(向量数据库)信息列表
  模型 item_names = [1,2,3]  => {1模型:[{item_name:检索,score:0.9},{},{}],2模型,3模型}
  milvus_hybrid_search(item_names:list[str])->dict[str,list[dict[str,Any]]]:
  4.1 获取milvus的客户端对象 infra
  4.2 准备存储最终结果的容器 {}
  4.3 循环item_names获取每个模型识别到item_name
      4.4  item_name生成稠密和稀疏向量
      4.5  创建item_name稠密和稀疏向量对应AnnSearchRequests列表 [milvus/infra]
      4.6  调用混合检索方法获取结果
      4.7  结果转成 [[ {id/pk:主键,distance:分,entity:{item_name:查询到item_name}} ]] ->  [{item_name:检索,score:0.9},{},{}]
      4.8  临时数据列表添加到字典 {}[item_name->模型遍历] = [{item_name:检索,score:0.9},{},{}] 查询匹配的列表
  4.9 返回最终结果 {}
"""
# 1. 获取数据和参数校验(session_id/original_query)
def validate_and_get_data(state:QueryGraphState) -> tuple[str,str]:
    # 1.1 state获取数据
    session_id:str = state.get("session_id")
    original_query = state.get("original_query")
    # 1.2 非空校验  空 -> ValueError / 非空
    if not session_id or not original_query:
        logger.error(f"session_id or original_query 为空,业务无法继续,提前终止!!")
        raise ValueError(f"session_id or original_query 为空,业务无法继续,提前终止!!")
    # 1.3 非空顺序返回值即可
    return session_id,original_query

# 2. 获取[有效的]聊天记录信息(session_id)
def valid_chat_message_list(session_id:str) -> list[dict]:
    # 2.1 调infra查询近期聊天数量 = 20  list[dict] -> {_id,role = "user / assistant",text,rewritten_query,item_names,image_urls,ts}
    recent_message_list:list[dict] = history_repository.list_recent(session_id=session_id,limit=20)
    # 2.2 过滤有效的聊天记录 条件1: 必须有item_names | 条件2: 过滤 role = user
    valid_message_list:list[dict] = [
        item
        for item in recent_message_list if item.get("item_names") and item.get("role") == 'user'
    ]
    # 2.3 返回有效的聊天记录
    logger.info(f"查询近期聊天数据共计:{len(recent_message_list)},过滤筛选后有效数据为:{len(valid_message_list)}")
    return valid_message_list


def filter_valid_chat_messages(message_list: list[dict]) -> list[dict]:
    """从入口节点已经加载的历史中筛选可用于主体识别的用户消息。"""
    return [
        item
        for item in message_list
        if item.get("item_names") and item.get("role") == "user"
    ]


def build_history_text(message_list: list[dict]) -> str:
    """将历史消息格式化为可直接放入提示词的文本。"""
    if not message_list:
        return "没有有效聊天记录!!"

    return "".join(
        f"{index},原始问题: {message.get('text')},"
        f"重写后的问题: {message.get('rewritten_query')}, "
        f"关联的item_name:{message.get('item_names')}\n"
        for index, message in enumerate(message_list, start=1)
    )


# 3.拼接提示词调用模型一次性识别item_names(不是最终)和重写问题(返回结构化数据json)
def call_llm_confirm_item_name_and_rewritten_query(message_list:list[dict],original_query:str) -> dict[str,str]:
    # 3.1 infra获取模型的客户端对象(json_mode)
    llm_client = llm_provider.chat(json_model=True)
    # 3.2 准备提示词需要的前置参数(query / history_text  "没有有效聊天记录!!" <-  <- message_list  index,原始问题: text,重写后的问题: rewritten_query, 关联的item_name:[1,2,3]\n )
    query = original_query
    history_text = build_history_text(message_list)
    # 3.3 加载提示词 shared/load...
    prompt_text:str = load_prompt("rewritten_query_and_itemnames",history_text = history_text,query = query)
    # 3.4 创建userMessage [ HumanMessage(content="xx") ]
    messages = [HumanMessage(content=prompt_text)]
    # 3.5 创建调用链 chains = llm | JSONOutputParser()
    chains = llm_client | JsonOutputParser()
    # 3.6 执行调用链 dict =  chains.invoke(xx)
    raw_result = chains.invoke(messages)
    if isinstance(raw_result, list) and all(isinstance(item, str) for item in raw_result):
        result = {"item_names": raw_result}
    else:
        result = require_json_object(raw_result)
    # 3.7 数据检验 if "item_names" not int dict: .... []
    if "rewritten_query" not in result:
        logger.warning(f"模型重写问题失败,给rewritten_query赋予原始问题:{original_query}")
        result["rewritten_query"] = original_query

        # 兜底：模型没识别出商品 → 设为空列表
    if "item_names" not in result:
        logger.warning("模型识别商品失败,给item_names赋予空列表")
        result["item_names"] = []
    elif not isinstance(result["item_names"], list) or not all(
        isinstance(item, str) for item in result["item_names"]
    ):
        logger.warning("模型识别的item_names格式无效,给item_names赋予空列表")
        result["item_names"] = []

    result["item_names"] = preserve_explicit_model_identifiers(result["item_names"], original_query)
    result["rewritten_query"] = enrich_rewritten_query(
        original_query, result["rewritten_query"]
    )
    # 3.8 返回结果dict
    logger.info(f"完成本次主体识别和问题重写,结果为:{result}")
    return result

# 4. 使用item_names进行向量数据库数据查询,获得每个item_name(模型)真实关联向量数据库的item_name(向量数据库)信息列表
# 模型 item_names = [1,2,3]  => {1模型:[{item_name:检索,score:0.9},{},{}],2模型,3模型}
def milvus_hybrid_search(item_names:list[str])->dict[str,list[dict[str,Any]]]:
    # 4.1 获取milvus的客户端对象 infra
    milvus_client = milvus_gateway.client
    # 4.2 准备存储最终结果的容器 {}
    milvus_search_result:dict[str,list[dict[str,Any]]] = {}
    # 4.3 循环item_names获取每个模型识别到item_name
    for item_name_llm in item_names:
        # 4.4  item_name生成稠密和稀疏向量
        result = llm_provider.generate_infra_embeddings([item_name_llm])
        dense_vector = result['dense'][0]
        sparse_vector = result['sparse'][0]
        # 4.5  创建item_name稠密和稀疏向量对应AnnSearchRequests列表 [milvus/infra]
        reqs:list[AnnSearchRequest] = milvus_gateway.create_requests(dense_vector=dense_vector,sparse_vector=sparse_vector,limit=5*2)
        # 4.6  调用混合检索方法获取结果
        milvus_result = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.chunks_collection,
            reqs = reqs,
            ranker_weights=(0.5,0.5),
            norm_score=True,
            limit=5
        )
        # 4.7  结果转成 [[ {id/pk:主键,distance:分,entity:{item_name:查询到item_name}} ]] ->  [{item_name:检索,score:0.9},{},{}]
        if milvus_result and milvus_result[0]:
            milvus_item_name_list:list[dict[str,Any]] = [
                {"item_name": item.get("entity",{}).get("item_name"),"score": item.get("distance",0.0)}
                for item in  milvus_result[0]
            ]
            logger.debug(f"大模型识别item_name:{item_name_llm},查询出关联的数据:{milvus_item_name_list}")
            # 4.8  临时数据列表添加到字典 {}[item_name->模型遍历] = [{item_name:检索,score:0.9},{},{}] 查询匹配的列表
            milvus_search_result[item_name_llm] = milvus_item_name_list
        else:
            logger.warning(f"大模型识别item_name:{item_name_llm},没有查询到任何相关数据!")
    # 4.9 返回最终结果 {}
    return milvus_search_result


def select_item_names(milvus_result:dict[str,list[dict[str,Any]]]) -> dict[str,list[str]]:
    # 5.2 提前定义两个列表 确认列表 list:[str]  可选列表 list:[str]
    confirmed_item_name_list:list[str] = []
    optional_item_name_list:list[str]  = []
    # 5.3 循环查询出来的字典数据  for _ , list in  dict.items()
    #说明循环对象每次会返回两个值，,_说明这个值接受但不使用
    for _ , milvus_item_list in milvus_result.items():
        # milvus_item_list.sort(key=lambda i: i.get("score",0.0),reverse=True)
        #  5.4 列表推导式 推导出大于0.65 [确定的阈值] -> 业务真实数据集测试
        confirmed_list:list[dict] = [
            item
            for item in milvus_item_list if item.get("score",0.0) > ITEM_NAME_CONFIRM_THRESHOLD
        ]
        #  5.5 判断是否存在大于 0.65数据,有 截取1 [0],添加到确定列表 continue;
        if confirmed_list:
            confirmed_item_name_list.append(confirmed_list[0].get("item_name"))
            continue
        # 5.6 继续列表推导式,推大于  0.5 < x < 0.65
        optional_list: list[dict] = [
            item
            for item in milvus_item_list if ITEM_NAME_CANDIDATE_THRESHOLD < item.get("score", 0.0) < ITEM_NAME_CONFIRM_THRESHOLD
        ]
        # 5.7 判断是否存在大于 0.5 小于 0.65的数据, 有截取[:2] , 添加可选列表 continue;
        if optional_list:
            optional_item_name_list.extend([ item.get("item_name") for item in  optional_list[:ITEM_NAME_OPTIONS_TOPK]] )
            continue
    # 5.8 返回两个列表 return {"确定":list,"可选":list}
    return {
        "confirmed_item_name_list":confirmed_item_name_list,
        "optional_item_name_list":optional_item_name_list
    }


#  6. 进行state数据修改 确定列表 -> rewritten_query | item_names 一定不能改answer return  可选有: answer = 有...确认!  return  answer = 完全没有...
def update_query_state(item_list:dict[str,list[str]],rewritten_query:str,state:QueryGraphState):
    # item_names + rewritten_query  ||  answer  有可选列表  ||  answer 没有任何 ...
    # 6.1 获取确认和可选列表
    confirmed_item_name_list:list[str] = item_list.get("confirmed_item_name_list")
    optional_item_name_list:list[str] = item_list.get("optional_item_name_list")
    #  6.2 确定列表不为空
    if confirmed_item_name_list:
        # 6.3 更新state item_names -> 确认列表  rewritten_query -> rewritten_query return
        state['item_names'] = confirmed_item_name_list
        state['rewritten_query'] = rewritten_query
        return
    # 6.4 可选列表不为空
    if optional_item_name_list:
        # 6.5 更新state answer -> 没有明确的主体名,但是有相似的: 1,2,3,4,请您确定后,再次提问题! return
        state['answer'] = f"没有明确的主体名,但是有相似的:{','.join(optional_item_name_list)},请您确定后,再次提问题!"
        return
    # 6.6 没有任何可信的和可选的
    # 6.7 更新state answer -> 没有任何可匹配的主体名,请确认后,再次提问!
    state['answer'] = f"没有任何可匹配的主体名,请确认后,再次提问!"

#  7. 保存user的聊天记录(所有都保存)
def save_user_chat_message(session_id,original_query,rewritten_query,item_names):
    # 7.1 调用mongodb infra进行存储
    history_repository.save_message(
        session_id=session_id,
        role="user",
        text=original_query,
        rewritten_query=rewritten_query,
        item_names=item_names,
        image_urls=[]
    )
    # 7.2 打印日志
    logger.info(f"完成{  '有效' if item_names else '无效' } 聊天记录保存!!")


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    # 1.校验和参数获取
    session_id,original_query = validate_and_get_data(state)
    # Resolver 已从历史确认主体时直接复用，不再调用 LLM、Embedding 和 Milvus。
    if state.get("reference_status") == "resolved" and state.get("item_names"):
        rewritten_query = state.get("rewritten_query") or original_query
        save_user_chat_message(session_id, original_query, rewritten_query, state["item_names"])
        return state

    # 2. 复用入口节点已加载的历史，避免再次访问 MongoDB。
    valid_message_list: list[dict] = filter_valid_chat_messages(state.get("history", []))
    # 3. 调用模型进行主体识别和问题重写
    result = call_llm_confirm_item_name_and_rewritten_query(valid_message_list,original_query)
    # 4. 进行向量数据库查询
    milvus_search_result: dict[str, list[dict[str, Any]]] = milvus_hybrid_search(item_names=result.get("item_names"))
    # 5. 获取确定和可选的数据列表
    item_name_dict:dict[str,list[str]] = select_item_names(milvus_search_result)
    # 6. 修改state数据
    update_query_state(item_name_dict,result.get("rewritten_query"),state)
    # 7. 保存聊天记录
    save_user_chat_message(session_id,original_query,result.get("rewritten_query"),state.get("item_names",[]))
    # 8. 返回state
    return state
