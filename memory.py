from typing import Dict, List

class ConversationMemory:
    def __init__(self, max_messages: int = 12):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_messages = max_messages

    def get_history(self, session_id: str):
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions.setdefault(session_id, []).append({"role": role, "content": content})
        self.sessions[session_id] = self.sessions[session_id][-self.max_messages:]

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)

memory = ConversationMemory()
