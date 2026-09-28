"""Local-development HTTP adapter. Use a real identity layer before deployment."""
from typing import Literal
from fastapi import FastAPI,HTTPException,Header
from pydantic import BaseModel,Field
from vitamind.mira.service import MiraSessionService
from vitamind.mira.reporting import reply_payload as payload,session_payload

app=FastAPI(title='Mira integrated assessment prototype',version='2.0.0')
sessions=MiraSessionService()

class StartSessionRequest(BaseModel):
    language:Literal['en','ar']

class MessageRequest(BaseModel):
    text:str=Field(min_length=1,max_length=10000)
    language:Literal['en','ar']

def lookup(session_id,token):
    try: return sessions.get(session_id,token)
    except KeyError: raise HTTPException(404,'session not found')
    except PermissionError: raise HTTPException(403,'valid session token required')

@app.get('/health')
def health(): return {'status':'ok','service':'mira','version':app.version,'clinical_validation':False}

@app.post('/api/v1/mira/session')
def start(request:StartSessionRequest):
    session,reply=sessions.start(request.language)
    return payload(session,reply)|{'session_token':session.session_token}

@app.post('/api/v1/mira/session/{session_id}/message')
def message(session_id:str,request:MessageRequest,x_mira_session_token:str|None=Header(default=None)):
    session=lookup(session_id,x_mira_session_token)
    if request.language!=session.language: raise HTTPException(400,'language does not match session')
    try: session,reply=sessions.message(session_id,request.text,x_mira_session_token)
    except ValueError as exc: raise HTTPException(400,str(exc))
    return payload(session,reply)

@app.get('/api/v1/mira/session/{session_id}')
def get(session_id:str,x_mira_session_token:str|None=Header(default=None)):
    s=lookup(session_id,x_mira_session_token)
    return session_payload(s)

@app.delete('/api/v1/mira/session/{session_id}')
def delete(session_id:str,x_mira_session_token:str|None=Header(default=None)):
    lookup(session_id,x_mira_session_token)
    sessions.delete(session_id,x_mira_session_token)
    return {'success':True}
