from app.shared.config.embedding_config import embedding_config, EmbeddingConfig
from app.shared.config.lm_config import lm_config, LLMConfig
from app.shared.config.bailian_mcp_config import mcp_config, McpConfig
from app.shared.config.milvus_config import milvus_config,MilvusConfig
from app.shared.config.mineru_config import mineru_config,MinerUConfig
from app.shared.config.minio_config import minio_config,MinIOConfig
from app.shared.config.reranker_config import reranker_config,RerankerConfig
from app.shared.config.settings_config import settings,AppSettings

from dataclasses import dataclass , field

#类->多个属性->每个属性承接一个配置对象
'''
    ！由于创建对象是引用类型，意思就是和shared.config中的类创建的对象指向同一块内存地址，
    所以在infra_config中修改属性值会影响shared.config中的对象属性值，会污染数据
    ！所以在infra_config中创建对象时，1.自己创建新的对象copy/new  2.使用field(default_factory=...)来创建新的对象实例，
    这样每次创建infra_config对象时，都会生成新的配置对象实例，避免污染数据。
    ！可以通过匿名函数lambda来创建新的对象实例，或者直接调用类名来创建新的对象实例。
    # @dataclass/pydantic BaseModel->直接给给对象赋默认值，默认值是引用类型（对象/字典/列表），都会报错
'''
@dataclass
class InfraConfig:
    #lambda函数创建新的没有参数的对象实例
    embedding_config:EmbeddingConfig=field(default_factory=lambda: embedding_config)
    lm_config:LLMConfig=field(default_factory=lambda: lm_config)
    mcp_config:McpConfig=field(default_factory=lambda: mcp_config)
    milvus_config:MilvusConfig=field(default_factory=lambda: milvus_config)
    mineru_config:MinerUConfig=field(default_factory=lambda: mineru_config)
    minio_config:MinIOConfig=field(default_factory=lambda: minio_config)
    reranker_config:RerankerConfig=field(default_factory=lambda: reranker_config)
    settings:AppSettings=field(default_factory=lambda: settings)

infra_config=InfraConfig()



