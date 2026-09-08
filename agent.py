import re
from typing import Dict, Any, List
from .llm import llm
from .memory import memory
from .rag import rag
from .tools import calculate_attendance

SYSTEM_PROMPT = """You are the AI Student Support Assistant.
Help students with academic and student-support questions.
You have access to academic document retrieval, calculation tools, and conversational memory.
Use supplied academic context when answering document-based questions.
Do not invent college rules. If information is unavailable, say it was not found in the knowledge base.
Explain calculations clearly and do not claim actions that were not performed.
"""

class StudentSupportAgent:
    def detect_intent(self, query: str) -> Dict[str, bool]:
        text = query.lower()
        calculation_terms = ["calculate", "percentage", "compute", "solve"]
        attendance_terms = ["attendance", "attended", "classes", "present"]
        academic_terms = ["syllabus", "subject", "semester", "regulation", "rule", "exam", "examination", "calendar", "timetable", "department", "college", "academic"]
        needs_tool = any(term in text for term in calculation_terms)
        needs_attendance = any(term in text for term in attendance_terms)
        needs_rag = any(term in text for term in academic_terms) or needs_attendance
        return {"needs_tool": needs_tool, "needs_attendance": needs_attendance, "needs_rag": needs_rag}

    @staticmethod
    def extract_attendance(query: str):
        patterns = [
            r"(\d+)\s*(?:out of|/)\s*(\d+)",
            r"(\d+)\s*(?:classes)?\s*attended.*?(\d+)\s*(?:classes)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1)), int(match.group(2))
        return None

    def execute(self, session_id: str, query: str) -> Dict[str, Any]:
        intent = self.detect_intent(query)
        tool_result = None
        if intent["needs_attendance"] or intent["needs_tool"]:
            attendance = self.extract_attendance(query)
            if attendance:
                try:
                    tool_result = {"tool": "attendance_calculator", "result": calculate_attendance(*attendance)}
                except ValueError as exc:
                    tool_result = {"tool": "attendance_calculator", "error": str(exc)}

        rag_result = rag.context(query) if (intent["needs_rag"] or not tool_result) else {"context": "", "sources": [], "results": []}
        history = memory.get_history(session_id)
        user_context = ""
        if history:
            user_context = "\nPrevious conversation:\n" + "\n".join(f"{x['role']}: {x['content']}" for x in history[-6:])
        prompt = f"""{SYSTEM_PROMPT}
Current user question:
{query}

Academic knowledge base context:
{rag_result['context']}

Tool execution result:
{tool_result}

{user_context}

Provide the best possible response. Use supplied context when relevant and never invent missing academic information.
"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        answer = llm.chat(messages)
        memory.add_message(session_id, "user", query)
        memory.add_message(session_id, "assistant", answer)
        return {"answer": answer, "intent": intent, "tool_result": tool_result, "sources": rag_result["sources"]}

agent = StudentSupportAgent()
