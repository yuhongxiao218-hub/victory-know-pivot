from mimetypes import guess_type
from pathlib import Path

from fastapi.responses import FileResponse, StreamingResponse
from fastapi import FastAPI, BackgroundTasks

from fastapi.requests import Request

from app.api.schema.query_schema import QueryRequest, QueryAsyncResponse, QueryResponse, HistoryDeleteResponse, \
    HistoryItemResponse, HistoryResponse
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import QueryGraphState, create_query_default_state
from app.rag.query.agent_trace_service import build_agent_trace
from app.shared.runtime.logger import logger
from app.shared.utils.path_util import PROJECT_ROOT
from app.shared.utils.sse_utils import sse_generator, push_to_session, create_sse_queue, SSEEvent
from app.shared.utils.task_utils import get_done_task_list, update_task_status, TASK_STATUS_COMPLETED, \
    TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, clear_task

app=FastAPI(description="查询页面的接口")

#接口1
@app.get("/html")
def return_chat_html():
    html_path_obj:Path=PROJECT_ROOT/"app"/"resources"/"html"/"chat.html"

    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj.name)[0]
    )


#接口2  状态检查
@app.get("/health")
def health():
    return {
        "code":0
    }

#接口3  流式接口

@app.get("/stream/{session_id}")
    #整个请求包装的对象， 可以获取数据 获取是否断开的状态
def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )



#接口4 问题触法接口 同步/异步
def invoke_query_graph(
    session_id,
    original_query,
    is_stream,
    debug=False,
    evaluation_mode=False,
):
    # 1. 准备state数据 [session_id,original_query,history = 省略]
    state: QueryGraphState = create_query_default_state(session_id=session_id,
                                                        original_query=original_query,
                                                        is_stream=is_stream)
    # 2. 动态执行(动态执行)
    try:
        # 任务的开始状态 task_utils
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        # result answer image_urls
        result = query_graph_app.invoke(state)
        agent_trace = (
            build_agent_trace(result, include_contexts=evaluation_mode)
            if debug or evaluation_mode
            else None
        )
        # 任务的成功状态 task_utils
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        if is_stream:
            if agent_trace is not None:
                push_to_session(session_id, SSEEvent.AGENT_TRACE, agent_trace)
            final_payload = {
                "answer": result.get("answer"),
                "status": "completed",
                "image_urls": result.get("image_urls"),
            }
            if agent_trace is not None:
                final_payload["agent_trace"] = agent_trace
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                final_payload,
            )
        return result

    except Exception as e:
        # 任务的失败状态 task_utils
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)  # 显示任务状态
        logger.exception(f"查询任务发生异常,异常信息为:{str(e)}")
        # progress  task_utils   ||\   delta  out...  final 输出结果..  error 推送错误的信息  -> sse_utils push_to_session
        # todo sse 多了error事件,向前端推送具体的信息文件!
        if is_stream:
            # {"error":"错误信息"}需要推送到前端跟用户实时报告
            push_to_session(session_id, SSEEvent.ERROR, {"error": f"异常信息为:{str(e)}"})#数据怎么传看前端 发现需要字典传递
        else:
            raise


@app.post("/query")
def query(tasks:BackgroundTasks,params:QueryRequest):

    # 本质就是执行查询图 [异步/同步]
    # 1.先获取参数 [session_id/original_query/is_stream]
    session_id = params.session_id
    original_query = params.query
    is_stream      = params.is_stream
    debug          = params.debug
    evaluation_mode = params.evaluation_mode

    # todo: 解决问题2   七个流程节点问完问题后流程直接完成，没有过度 清空task_utils
    clear_task(session_id)

    # 2.判断is_stream是否是异步
    if is_stream:
        # todo: 解决问题1
        create_sse_queue(session_id)
        # 是  -> tasks.add()
        tasks.add_task(
            invoke_query_graph,
            session_id=session_id,
            original_query=original_query,
            is_stream=is_stream,
            debug=debug,
            evaluation_mode=evaluation_mode,
        )
        # return QueryAsyncResponse
        return QueryAsyncResponse(
            message=f"{session_id}任务开始查询中...",
            session_id=session_id
        )
    else:
        # 不是 -> 调用方法
        result:QueryGraphState = invoke_query_graph(
            session_id=session_id,
            original_query=original_query,
            is_stream=is_stream,
            debug=debug,
            evaluation_mode=evaluation_mode,
        )
        # return QueryResponse
        # answer
        answer:str = result.get("answer")
        # image_urls
        image_urls:list[str] = result.get("image_urls",[])
        # done_list  ->  task_utils  get...
        done_list:list[str] = get_done_task_list(session_id)

        return QueryResponse(
            message=f"{session_id}已经处理完毕了!",
            session_id=session_id,
            answer=answer,
            done_list=done_list,
            image_urls=image_urls,
            agent_trace=(
                build_agent_trace(result, include_contexts=evaluation_mode)
                if debug or evaluation_mode
                else None
            ),
        )


    return QueryAsyncResponse / QueryResponse

#接口五：清空历史聊天记录
@app.delete("/history/{session_id}", response_model=HistoryDeleteResponse)
def clear_history(session_id:str):
    deleted_count =  history_repository.clear_session(session_id=session_id)
    logger.info(f"完成{session_id}的历史记录清空,共计删除:{deleted_count}条!")
    return HistoryDeleteResponse(
        message=f"完成{session_id}的历史记录清空,共计删除:{deleted_count}条!",
        deleted_count=deleted_count
    )

# 接口6: 查询聊天记录
@app.get("/history/{session_id}")
def list_history(session_id:str):
    history_message:list[dict] = history_repository.list_recent(session_id=session_id,limit=20)
    item_list:list[HistoryItemResponse] = [
        HistoryItemResponse(
            id=str(message_dict.get("_id")) , # ObjectId -> str
            session_id = session_id,
            role=message_dict.get("role"),
            text=message_dict.get("text"),
            rewritten_query=message_dict.get("rewritten_query"),
            item_names = message_dict.get("item_names",[]),
            image_urls=message_dict.get("image_urls",[]),
            ts = message_dict.get("ts")
        )
        for message_dict in history_message
    ]
    logger.info(f"查询{session_id}对应的聊天记录,共查询到:{len(item_list)}条记录!")
    return HistoryResponse(
        session_id=session_id,
        items=item_list
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8001)
