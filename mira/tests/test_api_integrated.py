from fastapi.testclient import TestClient
from app import app

def start(language='en'):
 c=TestClient(app);p=c.post('/api/v1/mira/session',json={'language':language}).json()
 return c,p['session_id'],{'X-Mira-Session-Token':p['session_token']}

def test_other_client_requires_token_for_all_session_operations():
 c,sid,headers=start()
 assert c.get(f'/api/v1/mira/session/{sid}').status_code==403
 assert c.delete(f'/api/v1/mira/session/{sid}').status_code==403
 assert c.post(f'/api/v1/mira/session/{sid}/message',json={'text':'hello','language':'en'}).status_code==403
 assert c.get(f'/api/v1/mira/session/{sid}',headers=headers).status_code==200
 assert c.delete(f'/api/v1/mira/session/{sid}',headers=headers).status_code==200

def test_api_crisis_followup_and_metadata():
 c,sid,h=start()
 for text in ['I want to kill myself.','Please stay with me.']:
  r=c.post(f'/api/v1/mira/session/{sid}/message',headers=h,json={'text':text,'language':'en'})
  assert r.status_code==200
  p=r.json();assert p['input_enabled'] and not p['assessment_complete']
  assert p['safety']['level']=='urgent'

def test_api_uses_model_and_preserves_language_contract():
 c,sid,h=start('ar')
 p=c.post(f'/api/v1/mira/session/{sid}/message',headers=h,json={'text':'أنا مشتت منذ الطفولة','language':'ar'}).json()
 assert p['model_assessment']['status']=='available'
 assert p['language']=='ar'
 assert c.post(f'/api/v1/mira/session/{sid}/message',headers=h,json={'text':'hello','language':'en'}).status_code==400

def test_invalid_language_and_empty_input():
 c=TestClient(app)
 assert c.post('/api/v1/mira/session',json={'language':'fr'}).status_code==422
 c,sid,h=start()
 assert c.post(f'/api/v1/mira/session/{sid}/message',headers=h,json={'text':'','language':'en'}).status_code==422
 assert c.post(f'/api/v1/mira/session/{sid}/message',headers=h,json={'text':'   ','language':'en'}).status_code==400
