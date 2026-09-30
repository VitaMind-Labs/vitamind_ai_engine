"""Optional local calendar. Parameterized SQL; patient-scoped reads; atomic commits."""
from pathlib import Path
import hashlib,json,sqlite3
from contextlib import contextmanager
from datetime import datetime,timezone
from .adhd.executive_function.schemas import Task,Outcome,OrganizeResponse
from .adhd.executive_function.errors import AssistantError

class LocalCalendar:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks(patient_id TEXT NOT NULL, task_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(patient_id,task_id));
            CREATE TABLE IF NOT EXISTS outcomes(patient_id TEXT NOT NULL, attempt_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(patient_id,attempt_id));
            CREATE TABLE IF NOT EXISTS requests(patient_id TEXT NOT NULL, request_id TEXT NOT NULL, request_hash TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(patient_id,request_id));
            ''')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db: yield db
        finally:
            db.close()
    def request_hash(self,request):
        raw=json.dumps(request.model_dump(mode='json'),sort_keys=True,separators=(',',':'))
        return hashlib.sha256(raw.encode()).hexdigest()
    def cached(self,request):
        with self.connect() as db:
            row=db.execute('SELECT request_hash,response FROM requests WHERE patient_id=? AND request_id=?',(str(request.patient.id),str(request.requestId))).fetchone()
        if row:
            if row[0]!=self.request_hash(request): raise AssistantError('INVALID_REQUEST','requestId was already used for a different payload',409)
            return OrganizeResponse.model_validate_json(row[1])
    def tasks(self,patient_id):
        with self.connect() as db:
            rows=db.execute('SELECT payload FROM tasks WHERE patient_id=? ORDER BY rowid',(str(patient_id),)).fetchall()
        return [Task.model_validate_json(r[0]).model_copy(update={'source':'CALENDAR'}) for r in rows]
    def outcomes(self,patient_id):
        with self.connect() as db:
            rows=db.execute('SELECT payload FROM outcomes WHERE patient_id=? ORDER BY rowid DESC LIMIT 1000',(str(patient_id),)).fetchall()
        return [Outcome.model_validate_json(r[0]) for r in rows]
    def commit(self,request,response):
        patient=str(request.patient.id)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            existing=db.execute('SELECT request_hash,response FROM requests WHERE patient_id=? AND request_id=?',(patient,str(request.requestId))).fetchone()
            if existing:
                if existing[0]!=self.request_hash(request): raise AssistantError('INVALID_REQUEST','requestId conflicts with a committed request',409)
                return OrganizeResponse.model_validate_json(existing[1])
            for op in response.taskOperations:
                if op.operation in ('ADD','RESCHEDULE') and op.task:
                    db.execute('INSERT INTO tasks VALUES(?,?,?) ON CONFLICT(patient_id,task_id) DO UPDATE SET payload=excluded.payload',(patient,op.task.temporaryId,op.task.model_dump_json()))
                    op.status='APPLIED'
                elif op.operation=='COMPLETE' and op.taskId:
                    row=db.execute('SELECT payload FROM tasks WHERE patient_id=? AND task_id=?',(patient,op.taskId)).fetchone()
                    if row:
                        task=Task.model_validate_json(row[0]); task.status='DONE'
                        db.execute('UPDATE tasks SET payload=? WHERE patient_id=? AND task_id=?',(task.model_dump_json(),patient,op.taskId)); op.status='APPLIED'
            for outcome in request.outcomes:
                key=(patient,str(outcome.attemptId))
                old=db.execute('SELECT payload FROM outcomes WHERE patient_id=? AND attempt_id=?',key).fetchone()
                if old and Outcome.model_validate_json(old[0])!=outcome:
                    raise AssistantError('INVALID_REQUEST','attemptId was already recorded with different outcome data',409)
                db.execute('INSERT OR IGNORE INTO outcomes VALUES(?,?,?)',(*key,outcome.model_dump_json()))
            db.execute('INSERT INTO requests VALUES(?,?,?,?)',(patient,str(request.requestId),self.request_hash(request),response.model_dump_json()))
        return response
