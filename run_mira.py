"""Run the integrated English/Arabic assessment locally, without an API key."""
import argparse
import sys
from vitamind.mira.agent import MiraAgent

def main():
    parser=argparse.ArgumentParser(description='Mira: integrated local screening conversation')
    parser.add_argument('--language',choices=['en','ar'],default='en')
    parser.add_argument('--demo',action='store_true',help='Run a fictional attention-related interview')
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    agent=MiraAgent()
    try: bundle=agent.model.classifier._load()
    except (ImportError,OSError,ValueError) as exc:
        parser.exit(1,f'Model could not load ({type(exc).__name__}). Run: python -m pip install -r requirements.txt\n')
    session,reply=agent.create_session(language=args.language)
    print('MIRA — local assessment prototype / تقييم أولي محلي')
    languages=(bundle.get('metadata') or {}).get('languages',[])
    if languages:
        print('Loaded model languages / لغات النموذج:', ', '.join(languages))
    print('Automatic report after at most 10 assessment answers / تقرير تلقائي بعد 10 إجابات تقييم كحد أقصى')
    print('Type exit to quit. Commands: report / تقرير, summary / ملخص, why / لماذا, skip / تخطي.\n')
    print('MIRA:',reply.text)
    demo={'concern':('I have been distracted since childhood and I forget appointments.','انا مشتت منذ الطفولة وانسي المواعيد'),
          'duration':('This has happened for years.','يحدث هذا منذ سنوات'),
          'impairment':('yes','نعم'),'settings':('yes','نعم'),'childhood':('yes','نعم'),
          'attention':('yes','نعم'),'confounders':('no','لا'),'sleep_need':('no','لا'),
          'energy':('no','لا'),'episodes':('no','لا'),'voices':('no','لا'),'beliefs':('no','لا')}
    turns=0
    while True:
        if args.demo:
            if reply.assessment_complete or turns>=24: break
            text=demo.get(session.pending.key,('skip','تخطي'))[args.language=='ar']
            print('YOU:',text)
        else:
            try: text=input('YOU: ').strip()
            except (EOFError,KeyboardInterrupt): break
            if text.lower() in {'exit','quit','خروج'}: break
            if not text: continue
        reply=agent.respond(session,text)
        turns+=1
        print('MIRA:',reply.text,'\n')
    print('Session closed / انتهت الجلسة')

if __name__=='__main__': main()
