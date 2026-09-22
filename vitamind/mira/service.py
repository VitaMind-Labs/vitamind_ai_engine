"""Session lifecycle boundary for HTTP adapters."""

from threading import Lock

from .agent import MiraAgent, MiraSession


class MiraSessionService:
    def __init__(self, agent: MiraAgent | None = None):
        self.agent = agent or MiraAgent()
        self._sessions: dict[str, MiraSession] = {}
        self._lock = Lock()

    def start(self, language: str = "en"):
        if language not in {"en", "ar"}:
            raise ValueError("language must be 'en' or 'ar'")
        session, reply = self.agent.create_session(language=language)
        with self._lock:
            self._sessions[session.session_id] = session
        return session, reply

    def get(self, session_id: str) -> MiraSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def message(self, session_id: str, text: str):
        session = self.get(session_id)
        return session, self.agent.respond(session, text)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None