# MinerU 模型版本配置（vlm = 视觉语言模型，适合PDF/图片高精度解析）
MINERU_MODEL_VERSION = "vlm"

# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
# 600 -> 一个pdf 约等于 1秒
MINERU_POLL_TIMEOUT_SECONDS = 600#轮询最大等待时间minerU

# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS = 3#一次轮询失败了，别立即开启下次！间隔等待

# MinerU 文件下载/单次请求超时时间（单位：秒），单次 HTTP 请求超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30


#day04里用到的：图片上下文的长度，约定100字符
IMAGE_CONTEXT_CHAR_LENGTH=100

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000

# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600

# 文本块重叠长度：相邻切块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50

# 最小拆分阈值：低于这个长度判定为无需拆分，直接尝试合并
CHUNK_MIN = 400

ITEM_NAME_RECOGNITION_CHUNKS_TOP_K=5

EMBEDDING_BATCH_SIZE=5

# 主体名称确认阈值：高于该分数 → 直接确认 -> 1
ITEM_NAME_CONFIRM_THRESHOLD = 0.65
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.50
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2