"""
Product:HiBug-6B-v1.0
chat前端调用格式
curl -X POST "http://127.0.0.1:8020/API" -H 'Content-Type: application/json' -d '{"prompt": "给一个冒泡排序代码"}'
code前端调用格式
curl -X POST "http://127.0.0.1:8020/api/code-gen" -H 'Content-Type: application/json' -d '{"text": "void do_test_creation()\r\n{\r\n    test_value = 0;\r\n    boost::thread thrd(&simple_thread);\r\n    thrd.join();\r\n    BOOST_CHECK_EQUAL(test_value, 999);\r\n}","language":"中文","category":"debug","program_lang":"python"}'
include：text/category/program_lang/language ect.
清除历史记录API:clear; 
STOP API：cancel;
设置历史信息使用history参数
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
import random
import threading
import json
import sys
from chains.filterbase import NaiveFilter,arraybase,arraymodel,arrayself,LANGUAGE_TAG


def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = getLogger('HiBugLog', 'hibug.log')


nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True



#environment
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

DEVICE2 = "cuda"
DEVICE_ID2 = "1"
CUDA_DEVICE2 = f"{DEVICE2}:{DEVICE_ID2}" if DEVICE_ID2 else DEVICE2


def is_in(full_str, sub_str):
    try:
        full_str.index(sub_str)
        return True
    except ValueError:
        return False
    
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def torch_gc_gen():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE2):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def predict_stream(inputs, modelcode, tokenizercode):
    n_token_prompt = len(inputs[0])
    response = "void"
    for tokens in modelcode.stream_generate(inputs,max_new_tokens=1024,top_k=1):
        response = tokenizercode.decode(tokens[0][n_token_prompt:])
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        yield json.dumps({
            'response': response,
            'code': 200,
            'time': time
        })
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    log = "[" + time + "] " + '", response:"' + repr(response) + '"'
    print(log)
    return torch_gc_gen()

def predict(text):
    words = text.split(',')  # 使用 split() 方法将字符串拆分成单词列表
    response = "void"
    for i in range(len(words)):
        response = ','.join(words[:i+1])  # 逐步生成不断增加的单词序列
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        yield json.dumps({
            'response': response,
            'code': 200,
            'time': time
        })
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    log = "[" + time + "] " + '", response:"' + repr(response) + '"'
    print(log)
    return torch_gc()


app = FastAPI()
file = NaiveFilter()
file.parse("sensitivewords.txt")

#previous_request: Optional[Future] = None
#lock = threading.Lock()
@app.post("/api/hibug-llm/chat")
async def create_item(request: Request):
    global hibug_model, hibug_doc, vs_path, file
    uid = 'void'
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    question = json_post_list.get('prompt')
    his = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p') 
    temperature = json_post_list.get('temperature')
    uid = json_post_list.get('uid')
    if question == None:
        logger.error(f"uid:{uid}, error: {'Prompt Miss a Value'}")
        return {
            'message': "Prompt Miss a Value",
            'code': 400
        } 
    try:
        if is_in(file.filtersen(question),"[[&]]"):
            res = "你好,你的输入含敏感词汇,请重新输入。" 
            res = predict(res)
            return EventSourceResponse(res)
        else:
            resp = hibug_doc.get_knowledge_based_answer(
                    query=question, 
                    vs_path=vs_path, 
                    chat_history=his if his else hibug_doc.history, 
                    streaming=True
            )
            return EventSourceResponse(resp)
    except Exception as e:
        logger.error(f"uid:{uid}, error: {e}")
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            'message': "Chat Model Service Failure",
            'code': 400,
            'time': time
        }      

@app.post("/api/hibug-llm/code-gen")
async def create_item(request: Request):
    global hibug_model, hibug_doc, vs_path, modelcode, tokenizercode,file
    uid = 'void'
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    his = json_post_list.get('history')   
    text = json_post_list.get('text')
    category = json_post_list.get('category')
    language = json_post_list.get('language')
    program_lang = json_post_list.get('program_lang')
    uid = json_post_list.get('uid')
    if text == None or language == None or program_lang == None:
        logger.error(f"uid:{uid}, error: {'Parameter Miss a Value(text or language or program_lang)'}")
        return {
            'message': "Parameter Miss a Value(text or language or program_lang)",
            'code': 400
        } 
    try:
        if category == "generation":
            prompt = '#' + LANGUAGE_TAG[program_lang] + '#' + text
            #print(prompt)
            inputs = tokenizercode.encode(prompt, return_tensors="pt").to(modelcode.device)
            res = predict_stream(inputs, modelcode, tokenizercode)
            return EventSourceResponse(res)
        else:
            if  category=="completion" and language=="中文":
                prompt = "以下是部分" + program_lang +"代码，请对代码进行补全：\n" + text
            elif  category=="completion" and language=="English":
                prompt = "Please complete the following " + program_lang + " code to form a complete piece of code:\n" + text
            # 生成注释
            elif category=="comment" and language=="中文":
                prompt = "请为下列代码生成注释，使用中文：\n" + text
            elif category=="comment" and language=="English":
                prompt = "Please generate comments for the following code using English:\n" + text
            # debug
            elif category=="debug" and language=="中文":
                prompt = "请为下列"+ program_lang +"代码进行debug，指出错误，并给出修改后的代码，使用中文：\n" + text
            elif category=="debug" and language=="English":
                prompt = "Please debug the following "+ program_lang +" code and provide the modified code using English:\n" + text
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
    except Exception as e:
        logger.error(f"uid:{uid}, error: {e}")
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            'message': "Code-Gen Model Service Failure",
            'code': 400,
            'time': time
        }

@app.get("/api/hibug-llm/cancel")
async def cancel():
    global hibug_doc
    try:
        # 首先尝试获取锁，如果锁被其他线程占用，则说明有其他线程正在运行，触发取消操作
        if hibug_doc.isprocess:
            hibug_doc.cancel_sse = True
            now = datetime.datetime.now()
            time = now.strftime("%Y-%m-%d %H:%M:%S")
            return {
                'message': "Request successfully cancelled",
                'code': 200,
                'time': time
            }
        else:
            now = datetime.datetime.now()
            time = now.strftime("%Y-%m-%d %H:%M:%S")
            return {
                'message': "Failed: There is no content being output, no need to stop!",
                'time': time,
                'code': 400
            }
    except Exception as e:
        logger.error(f"error: {e}")
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            'message': "Stop Output Service Failure",
            'code': 400,
            'time': time
        }

@app.get("/api/hibug-llm/clear")
async def clear():
    global hibug_doc
    try:
        hibug_doc.history = []
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            'response': "History successfully cleared",
            'code': 200,
            'time': time
        }
    except Exception as e:
        logger.error(f"error: {e}")
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            'message': "Clear History Service Failure",
            'code': 400,
            'time': time
        }
    
if __name__ == '__main__':
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    try:
        shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
        hibug_model = shared.loaderLLM('HiBug', False, False)
        hibug_model.history_len = LLM_HISTORY_LEN
        hibug_doc = Hibugdoc()
        hibug_doc.init_cfg(llm_model=hibug_model,
                            embedding_model=EMBEDDING_MODEL,
                            embedding_device=EMBEDDING_DEVICE,
                            top_k=VECTOR_SEARCH_TOP_K)
        filepath = 'locallib'
        vs_path, _ = hibug_doc.init_knowledge_vector_store(filepath)
        tokenizercode = AutoTokenizer.from_pretrained("hibugcodemodel", trust_remote_code=True)
        modelcode = AutoModel.from_pretrained("hibugcodemodel", trust_remote_code=True,device='cuda:1')
        modelcode = modelcode.eval()
    except Exception as e:
        logger.error(f"error: {e}")
    uvicorn.run(app, host='0.0.0.0', port=8020, workers=1)
