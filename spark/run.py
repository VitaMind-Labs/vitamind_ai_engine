import argparse,json,sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4,UUID
from pydantic import ValidationError
from lumina.calendar import LocalCalendar
from lumina.adhd.executive_function.orchestrator import LuminaADHD
from lumina.adhd.executive_function.errors import AssistantError
from lumina.adhd.executive_function.schemas import OrganizeRequest

ROOT=Path(__file__).resolve().parent
DEFAULT_PATIENT=UUID('ae3e23cc-5b26-468f-a8df-708519a4a144')

def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='Offline Lumina ADHD AI; no graphical interface')
    sub=parser.add_subparsers(dest='command',required=True)
    analyze=sub.add_parser('analyze'); analyze.add_argument('file',type=Path); analyze.add_argument('--calendar',type=Path)
    interactive=sub.add_parser('interactive'); interactive.add_argument('--lang',choices=['EN','AR'],default='EN'); interactive.add_argument('--calendar',type=Path,default=ROOT/'runtime_data/calendar.sqlite3'); interactive.add_argument('--patient',type=UUID,default=DEFAULT_PATIENT)
    serve=sub.add_parser('serve'); serve.add_argument('--port',type=int,default=8011); serve.add_argument('--calendar',type=Path,default=ROOT/'runtime_data/calendar.sqlite3')
    args=parser.parse_args()
    try:
        if args.command=='serve':
            import uvicorn
            from lumina.api import create_app
            uvicorn.run(create_app(str(args.calendar)),host='127.0.0.1',port=args.port,access_log=False)
            return
        calendar=LocalCalendar(args.calendar) if args.calendar else None
        assistant=LuminaADHD(calendar)
        if args.command=='analyze':
            request=OrganizeRequest.model_validate_json(args.file.read_text(encoding='utf-8-sig'))
            print(assistant.organize(request).model_dump_json(indent=2)); return
        print('Lumina ADHD · local development assistant · /exit to stop · /json for full output')
        print(f'Calendar: {args.calendar}\nPatient: {args.patient}\nDate uses this computer\'s local clock. Tasks are saved locally when clear; no external calendar is contacted.')
        last=None
        while True:
            try: text=input('\nYou > ').strip()
            except EOFError: return
            if text=='/exit': return
            if text=='/json':
                if last: print(last.model_dump_json(indent=2))
                continue
            if not text: continue
            clock=datetime.now().astimezone()
            payload={'requestId':str(uuid4()),'patient':{'id':str(args.patient),'language':args.lang},'message':{'text':text},'timeContext':{'referenceDate':clock.date().isoformat(),'localTime':clock.time().replace(tzinfo=None).isoformat()},'calendar':{'commit':True}}
            last=assistant.organize(payload)
            print(last.response.text)
            print(f'[{last.intent} | capacity={last.capacity} | safety={last.safety.level}]')
            if any(op.status=='APPLIED' for op in last.taskOperations):
                print('Saved in local calendar.' if args.lang=='EN' else 'تم الحفظ في التقويم المحلي.')
    except (ValidationError,AssistantError,OSError,ValueError) as exc:
        print(f'Error: {exc}',file=sys.stderr); raise SystemExit(2)

if __name__=='__main__': main()
