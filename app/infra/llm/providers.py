from langchain_community.chat_models import ChatOpenAI

from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client, get_reranker_model

from app.shared.model.embedding_utils import generate_embeddings
class LLMProviders():
    """
    大语言模型会使用兜底策略：不传 model 时，get_llm_client 会自动回落到 lm_config.llm_model（文本大模型），正是 chat 想要的
    视觉语言模型：不能使用文本模型，所以需要作条件判断，不是视觉模型要更换，通过infra_config这个门面进行
    """
    #大语言模型
    def chat(self,model:str|None=None,json_model:bool=False)->ChatOpenAI:
        return get_llm_client(model=model,json_mode=json_model)
    #视觉模型
    def vision_chat(self,model:str|None=None)->ChatOpenAI:
        if not model:
            model=infra_config.lm_config.lv_model
        return get_llm_client(model=model,json_mode=False)#不传参json_mode，因为写死不需要json格式


    #--------后续修改拓展--------
    #嵌入式模型
    def generate_infra_embeddings(self,texts:list[str])->dict[str,list]:
        return  generate_embeddings(texts)
    #重排序模型
    def reranker_model(self):
        return get_reranker_model()



llm_provider=LLMProviders()