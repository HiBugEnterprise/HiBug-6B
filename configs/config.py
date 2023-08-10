"""
Product:HiBug1.0
"""
import torch.cuda
import torch.backends
import os
import uuid

embedding_model_dict = {
    "hibugembed": "./embedlib",
}

# Embedding model name
EMBEDDING_MODEL = "hibugembed"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

llm_model_dict = {
    "HiBug": {
        "name": "HiBug",
        "pretrained_model_name": "HiBug",
        "local_model_path": './hibugmodel',
        "provides": "Hibug"
    },
}
LLM_MODEL = "hibug"
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = True
LLM_LORA_PATH = ""
CACHED_VS_NUM = 10
SENTENCE_SIZE = 400
CHUNK_SIZE = 800
LLM_HISTORY_LEN = 5
VECTOR_SEARCH_TOP_K = 5
VECTOR_SEARCH_SCORE_THRESHOLD = 50
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
USE_LORA = True if LLM_LORA_PATH else False
USE_PTUNING_V2 = False
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业地来回答用户的问题。不允许在答案中添加编造成分。问题是：{question}"""
