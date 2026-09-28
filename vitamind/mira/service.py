"""Process-local demo sessions with per-session capability tokens and locking."""
from threading import RLock
import secrets
from .agent import MiraAgent

class MiraSessionService:
    def __init__(self,agent=None):
        self.agent=agent or MiraAgent()
        self._sessions={}
        self._lock=RLock()
    def start(self,language='en'):
        with self._lock:
            session,reply=self.agent.create_session(language=language)
            self._sessions[session.session_id]=session
            return session,reply
    def get(self,session_id,token=None):
        with self._lock:
            session=self._sessions.get(session_id)
            if session is None: raise KeyError(session_id)
            if not token or not secrets.compare_digest(token,session.session_token): raise PermissionError('Invalid session token')
            return session
    def message(self,session_id,text,token=None):
        with self._lock:
            session=self.get(session_id,token)
            return session,self.agent.respond(session,text)
    def delete(self,session_id,token=None):
        with self._lock:
            self.get(session_id,token)
            del self._sessions[session_id]
            return True
