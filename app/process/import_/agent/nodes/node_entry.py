from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.entry_service import resolve_input_file

#装饰器的作用：打印日志
#返回一个装饰器 deco，deco 再包装真正的节点函数。

@node_log("node_entry")
#节点函数（state是全局共享域，也就是runtime txt）
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """

    add_running_task(state["task_id"], "node_entry")
    #其实进行中的结点只有一个所以可以不用list
    """
    add_running_task、add_done_task来自util的工具类
    将进行中或者完成的结点存储到容器中-> 容器 <-fastapi结点读取<-轮询1s<-前端页面
    为什么有task_id? 底层存储使用字典，将task_id作为key也是客户端的唯一标识，每个客户端存储一个对应的列表list作为value
                   1.其实进行中的结点running_task用list，可能出现单个结点或者并行结点（不是指上传多个文件 而是指达到查询效果时可能会有多个线程并行）的情况
                   2.结束的结点done_task可以用list存储，因为可能有多个结点完成
    """
    # running 用 list 的原因：同一个 task 内可能同时有多个节点在跑。
    #   - 单链图（import）：同一时刻只有 1 个
    #   - 并行图（query 的多路搜索）：多个分支同时执行，就有多个
    # 注意：这不是「同时上传多个文件」——不同文件是不同 task_id，
    #       各自一个列表，互不干扰；这里说的是单个任务内部的并行。

    state = resolve_input_file(state)#结点调用业务函数来实现分离后的连接

    add_done_task(state["task_id"], "node_entry")

    return state


if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    from app.process.import_.agent.state import create_default_state
    import json
    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    result_1 =  node_entry(test_state1)
    print(f"第一次测试结果: \n {json.dumps(result_1, indent=4, ensure_ascii=False)}")
    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    result_2 = node_entry(test_state2)
    print(f"第二次测试结果: \n {json.dumps(result_2, indent=4, ensure_ascii=False)}")
    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    result_3 = node_entry(test_state3)

    print(f"第三次测试结果: \n {json.dumps(result_3, indent=4, ensure_ascii=False)}")

    logger.info("===== 结束node_entry节点单元测试 =====")