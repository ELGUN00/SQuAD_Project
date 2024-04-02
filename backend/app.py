from fastapi import Request, FastAPI

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import torch.nn.functional as F

import numpy as np


app = FastAPI()

model_name = "deepset/xlm-roberta-large-squad2"
# model = AutoModelForQuestionAnswering.from_pretrained('./model/model/',local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained('./model/tokenizer/',local_files_only=True)

@app.post("/predict")
async def get_body(request: Request):
    req = await request.json()
    if req['question'] == None or req['context']==None:
        return {'question':None,'asnwer':None,'score':None}
    
    inputs = tokenizer.encode_plus(req['question'], req['context'], add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    res = model(**inputs)

    answer_start = torch.argmax(
        res['start_logits'])
    
    answer_end = torch.argmax(res['end_logits']) + 1 
    start = F.softmax(res['start_logits'], dim=1)
    end = F.softmax(res['end_logits'], dim=1)

    start_index = torch.argmax((start), dim=1)
    end_index = torch.argmax((end), dim=1)
    
    outer = np.matmul(np.expand_dims(start.detach().numpy(), -1), np.expand_dims(end.detach().numpy(), 1))
    max_answer_len = 512
    candidates = np.tril(np.triu(outer), max_answer_len - 1)
    # idx_sorts = [np.argmax(candidates[i].flatten()) for i in range(len(candidates))]
    scores = [candidates[[i], start_index[i], end_index[i]][0] for i in range(len(candidates))]
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    res = {'question':req.get('question'),'asnwer':answer,'score':round(float(scores[0]),2)}
    return res

@app.get("/metrics")
async def get_body(request: Request):
    res = {'f1 score':0.66,'BLEU':0.52}
    return res