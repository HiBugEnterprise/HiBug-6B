"""
Product:HiBug-6B-v1.0
前端调用格式
curl -X POST "http://127.0.0.1:8020/API" -H 'Content-Type: application/json' -d '{"prompt": "给一个冒泡排序代码"}'
para:prompt
清除历史记录clear=True; 设置历史信息使用history参数
vector history多轮，可调整进入LLM的轮数
约15GB显存
"""
from fastapi import FastAPI, Request
#from concurrent.futures import Future
#from typing import Optional
import asyncio
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import uvicorn, json, datetime
from typing import List, Optional
import torch
import os
from configs.config import *
from chains.hibugdata import Hibugdoc
import os
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
from sse_starlette.sse import EventSourceResponse
import json

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True



#environment
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()

#previous_request: Optional[Future] = None

@app.post("/api/chat")
async def create_item(request: Request):
    global hibug_model, hibug_doc, vs_path
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    question = json_post_list.get('prompt')
    his = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p') 
    temperature = json_post_list.get('temperature')
#    previous_request = asyncio.get_event_loop().create_future()
    # SSE流式输出
    resp = hibug_doc.get_knowledge_based_answer(
            query=question, 
            vs_path=vs_path, 
            chat_history=his if his else hibug_doc.history, 
            streaming=True
    )
    return EventSourceResponse(resp)

@app.post("/api/code-gen")
async def create_item(request: Request):
    global hibug_model, hibug_doc, vs_path
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    his = json_post_list.get('history')   
    text = json_post_list.get('text')
    category = json_post_list.get('category')
    language = json_post_list.get('language')
    program_lang = json_post_list.get('program_lang')
    # 生成代码
    if category=="generation"and language=="中文":
        prompt =  "请使用"+ language +"编程语言，" + text
    elif category=="generation"and language=="English":
        prompt = "Please use "+ language +" programming language, "+ text
    # 代码补全
    elif  category=="completion" and language=="中文":
        prompt = "以下是部分代码，请对代码进行补全：\n" + text
    elif  category=="completion" and language=="English":
        prompt = "Please complete the following code:\n" + text
    # 生成注释
    elif category=="comment" and language=="中文":
        prompt = "请为下列代码生成注释，使用中文：\n" + text
    elif category=="comment" and language=="English":
        prompt = "Please generate comments for the following code using English:\n" + text
    # debug
    elif category=="debug" and language=="中文":
        prompt = "请为下列代码debug调试，并给出修改后的代码，使用中文：\n" + text
    elif category=="debug" and language=="English":
        prompt = "Please debug the following code and provide the modified code using English:\n" + text
    else:
        prompt = text
#    previous_request = asyncio.get_event_loop().create_future()
    # SSE流式输出
    resp = hibug_doc.get_knowledge_based_answer(
            query=prompt, 
            vs_path=vs_path, 
            chat_history=his if his else hibug_doc.history, 
            streaming=True
    )
    return EventSourceResponse(resp)

@app.get("/cancel")
async def cancel():
    global hibug_doc
    hibug_doc.cancel_sse = True
    return{"message":"Request cancelled"}

@app.get("/clear")
async def clear():
    global hibug_doc
    hibug_doc.history = []
    return{"message":"Request cleared"}

if __name__ == '__main__':
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    hibug_model = shared.loaderLLM('HiBug', False, True)
    hibug_model.history_len = LLM_HISTORY_LEN
    hibug_doc = Hibugdoc()
    #print(hibug_doc.history)
    hibug_doc.init_cfg(llm_model=hibug_model,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    filepath = 'locallib'
    vs_path, _ = hibug_doc.init_knowledge_vector_store(filepath)

    uvicorn.run(app, host='0.0.0.0', port=8020, workers=1)
