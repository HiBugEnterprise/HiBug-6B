"""
Product:HiBug1.0
test
"""
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
from configs.config import *
from chains.hibugdata import Hibugdoc
import os
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

REPLY_WITH_SOURCE = True


def main():
    hibug_model = shared.loaderLLM('HiBug', False, True)
    hibug_model.history_len = LLM_HISTORY_LEN

    hibug_doc = Hibugdoc()
    hibug_doc.init_cfg(llm_model=hibug_model,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    vs_path = None
    while not vs_path:
        filepath = 'locallib'
        if not filepath:
            continue
        vs_path, _ = hibug_doc.init_knowledge_vector_store(filepath)
    history = []
    while True:
        query = input("（HiBug）请输入问题：")
        last_print_len = 0
        for resp, history in hibug_doc.get_knowledge_based_answer_demo(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,
                                                                     streaming=STREAMING):
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
        print(history)


if __name__ == "__main__":
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    main()
