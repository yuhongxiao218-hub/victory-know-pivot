"""
工具模块，负责提供 lm 相关的辅助能力。
"""
import json

from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI
#方式一：from langchain.chat_model import init_chat_model   这个init_chat_model适合多智能体的项目，未指明使用哪个provider
#方式二：from langchain_openai import ChatOpenAI   这个适合指定使用某个provider场景
#init_chat_model ->provider=openai ->ChatOpenAI

from app.shared.config.lm_config import lm_config
from app.shared.runtime.logger import logger

_DEFAULT_LLM_MODEL = "qwen3-32b"
_DEFAULT_TEMPERATURE = 0.1


def require_json_object(value):
    """接受模型偶尔返回的单元素对象数组，其余非对象结构直接拒绝。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, list):
        for message in reversed(value):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            parts = content if isinstance(content, list) else [content]
            for part in reversed(parts):
                text = part.get("text") if isinstance(part, dict) else part
                if not isinstance(text, str):
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
    raise ValueError("模型结构化输出必须是单个 JSON 对象")


#缓存模型的容器
"""
# 缓存模型的容器：dict[tuple[str, bool], ChatOpenAI]
# 获取视觉模型 / 大语言模型：调用后返回的数据类型1. 非格式化数据：普通文本2. 格式化数据：JSON
#True 和 False 表示：是否要求大语言模型以结构化 JSON 形式返回结果。
# 视觉模型没有格式化；大语言模型格式化或者不格式化。
# {
#     ("视觉模型的名字", False): 视觉模型对象，
#     ("大语言模型的名字", False): 大语言模型对象，
#     ("大语言模型的名字", True): 大语言模型对象
# } 所以其实有三个情况。
#
# 缓存 -> 实例化一次 -> 存储：单个变量->后续可以通过key进行多个变量的存储
"""
_llm_client_cache: dict[tuple[str, bool], ChatOpenAI] = {}


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取带“全局缓存”的LangChain ChatOpenAI客户端实例
    适配OpenAI/千问/即梦AI等**OpenAI兼容API**，支持自定义模型和JSON标准化输出
    核心特性：缓存机制+配置统一加载+异常精准捕获+国产模型参数适配

    :param model: 模型名称，优先级：传入参数 > 配置文件 lm_config.llm_model > 内置默认模型
    :param json_mode: 是否开启JSON输出模式，开启后返回标准json_object格式（适配结构化数据解析）
    :return: 初始化完成的ChatOpenAI实例（优先从全局缓存获取，未命中则新建并缓存）
    :raise ValueError: 缺失API密钥/基础地址等核心配置
    :raise Exception: 模型初始化失败（LangChain封装层异常）
    """
    # 1. 确定目标模型（优先级递减，保证模型名非空）
    #调用大语言模型可以不传入model或.env配置文件或者全局变量大预言模型
    target_model = model or lm_config.llm_model or _DEFAULT_LLM_MODEL
    # 缓存键：模型名+JSON模式，唯一标识不同配置的客户端
    cache_key = (target_model, json_mode)

    # 2. 缓存命中：直接返回已初始化的实例，避免重复创建
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. 核心配置校验：拦截缺失的API关键配置，提前抛出明确异常
    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_BASE_URL（API接口基础地址）")
    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    # 4. 配置参数组装：区分「国产模型私有参数」和「OpenAI通用参数」
    # extra_body：千问/即梦等国产模型专属私有参数（LangChain透传至API）
    extra_body = {"enable_thinking": False}  # 千问专属：关闭思考链输出，减少冗余内容
    # model_kwargs：OpenAI通用参数，所有兼容API均支持
    #是否开启思考模式
    model_kwargs = {}
    if json_mode:
        # 开启JSON标准输出模式，强制模型返回可解析的json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    # 5. 客户端初始化：捕获LangChain封装层异常，抛出更友好的提示
    try:
        llm_client = ChatOpenAI(
            model=target_model,  # 目标模型名
            temperature=lm_config.llm_temperature or _DEFAULT_TEMPERATURE,  # 低温度保证输出确定性（0~1）
            api_key=lm_config.api_key,  # API密钥
            base_url=lm_config.base_url,  # API基础地址（适配国产模型代理地址）
            timeout=60,
            max_retries=1,
            extra_body=extra_body,  # 国产模型私有参数透传
            model_kwargs=model_kwargs,  # OpenAI通用参数
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    # 6. 新实例存入全局缓存，供后续调用复用
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return llm_client

