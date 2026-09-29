"""Run from the lumina/ folder: python journal_ai/examples/integration.py"""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from journal_ai import JournalSentinel
from journal_ai.reporting import weekly_report

agent=JournalSentinel()
analyses=[
    agent.analyze("I feel anxious and overwhelmed today",lang="en",entry_id="demo-en"),
    agent.analyze("أشعر بالقلق اليوم والمهام متراكمة علي",lang="ar",entry_id="demo-ar"),
]
print(json.dumps({"analyses":analyses,"weekly_summary":weekly_report(analyses)},ensure_ascii=False,indent=2))
