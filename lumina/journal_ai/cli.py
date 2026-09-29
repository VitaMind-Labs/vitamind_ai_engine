import argparse
import json
import sys
from pathlib import Path
from .pipeline import JournalSentinel
from .reporting import weekly_report

def main():
    if hasattr(sys.stdin,"reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="Journal Sentinel: local English/Arabic AI, no interface or API key")
    sub=parser.add_subparsers(dest="command",required=True)
    analyze=sub.add_parser("analyze",help="Analyze journal text")
    group=analyze.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file",type=Path,help="UTF-8 text file")
    group.add_argument("--stdin",action="store_true",help="Read a JSON request from stdin")
    analyze.add_argument("--lang",choices=["en","ar"])
    analyze.add_argument("--entry-id",default="terminal-entry")
    analyze.add_argument("--audit",action="store_true")
    analyze.add_argument("--model-dir",type=Path)
    batch=sub.add_parser("batch",help="Analyze JSONL requests; one JSON result per line")
    batch.add_argument("file",type=Path)
    report=sub.add_parser("report",help="Summarize a JSON array of saved analyses")
    report.add_argument("file",type=Path)
    report.add_argument("--lang",choices=["en","ar"],default="en")
    sub.add_parser("interactive",help="Try journal entries in your terminal")
    args=parser.parse_args()
    try:
        if args.command=="report":
            result=weekly_report(json.loads(args.file.read_text(encoding="utf-8-sig")),lang=args.lang)
        elif args.command=="batch":
            agent=JournalSentinel()
            for line in args.file.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    print(json.dumps(agent.analyze(**json.loads(line)),ensure_ascii=False))
            return
        elif args.command=="interactive":
            agent=JournalSentinel()
            print("Journal Sentinel · local synthetic-data prototype · type /exit to stop")
            print("English or العربية. Entries are not saved or sent to anyone.")
            index=0
            while True:
                try:
                    text=input("\nJournal > ")
                except EOFError:
                    return
                if text.strip()=="/exit":
                    return
                if not text.strip():
                    continue
                index+=1
                result=agent.analyze(text,entry_id=f"terminal-{index}")
                print(result["response"]["text"])
                print(f"[tier={result['tier']} | action={result['action']['type']} | notification=not_sent]")
            return
        else:
            payload=json.load(sys.stdin) if args.stdin else {"text":args.text if args.text is not None else args.file.read_text(encoding="utf-8-sig"),"lang":args.lang,"entry_id":args.entry_id,"include_audit":args.audit}
            result=JournalSentinel(args.model_dir).analyze(**payload)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError, OSError, TypeError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        raise SystemExit(2)
