from typing import Any

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    session_id:str
    is_stream: bool
    debug: bool = False
    evaluation_mode: bool = False

#流式输出的response
class QueryAsyncResponse(BaseModel):
    message: str
    session_id:str

#非流式输出
class QueryResponse(BaseModel):
    message: str
    session_id: str
    answer:str
    done_list:list[str]
    image_urls:list[str]
    agent_trace: dict[str, Any] | None = None

class HistoryDeleteResponse(BaseModel):
    message:str
    deleted_count:int

class HistoryItemResponse(BaseModel):
    id:str
    session_id:str
    role:str
    text:str
    rewritten_query:str
    item_names:list[str]
    image_urls:list[str]
    ts:float#时间戳

class HistoryResponse(BaseModel):
    session_id:str
    items:list[HistoryItemResponse]

