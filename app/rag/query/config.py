# 主体名称确认阈值：高于该分数 → 直接确认 -> 1
ITEM_NAME_CONFIRM_THRESHOLD = 0.65
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.50
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2

# 默认返回的最大知识库片段数量
RETRIEVAL_DEFAULT_LIMIT = 5
# 混合检索权重：dense向量权重 0.9，sparse向量权重 0.1
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)