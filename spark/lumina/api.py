from contextlib import asynccontextmanager
import logging,os,sqlite3
from pathlib import Path
from time import perf_counter
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from .calendar import LocalCalendar
from .adhd.executive_function.orchestrator import LuminaADHD
from .adhd.executive_function.schemas import OrganizeRequest,OrganizeResponse
from .adhd.executive_function.errors import AssistantError

logger=logging.getLogger('lumina.adhd')
def create_app(calendar_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path=calendar_path or os.environ.get('LUMINA_CALENDAR_PATH')
        app.state.assistant=LuminaADHD(LocalCalendar(path) if path else None)
        yield
    app=FastAPI(title='Lumina ADHD AI',version='1.1.0',lifespan=lifespan)
    @app.exception_handler(RequestValidationError)
    async def invalid(request,exc):
        language=any('language' in e['loc'] for e in exc.errors())
        return JSONResponse(status_code=422,content={'errorCode':'UNSUPPORTED_LANGUAGE' if language else 'INVALID_REQUEST','message':'Request does not match the schema','fields':[{'location':list(e['loc']),'type':e['type']} for e in exc.errors()]})
    @app.exception_handler(AssistantError)
    async def typed_error(request,exc):
        return JSONResponse(status_code=exc.status,content={'errorCode':exc.code,'message':exc.message})
    @app.get('/health')
    def health(): return {'status':'ok','service':'lumina-adhd','version':'1.1.0'}
    @app.get('/ready')
    def ready():
        assistant=app.state.assistant
        ok=assistant.ready
        intent='LOCAL_TRAINED_CLASSIFIER' if assistant.intent_classifier.model.artifact else 'RULES_NO_TRAINED_INTENT_ARTIFACT'
        friction='LOCAL_TRAINED_CLASSIFIER' if assistant.friction_classifier.model.artifact else 'RULES_NO_TRAINED_FRICTION_ARTIFACT'
        return JSONResponse(status_code=200 if ok else 503,content={'ready':ok,'safetyModelLoaded':ok,'intent':intent,'friction':friction,'calendarEnabled':assistant.calendar is not None})
    @app.get('/version')
    def version():
        assistant=app.state.assistant
        return {'module':'1.1.0','schema':'1.0.0','rules':'1.0.0','templates':'1.0.0','intentModel':assistant.intent_classifier.model.metadata.get('modelVersion','rules'),'frictionModel':assistant.friction_classifier.model.metadata.get('modelVersion','rules'),'externalAI':False,'tokenUsage':None}
    @app.post('/v1/lumina/adhd/organize',response_model=OrganizeResponse)
    def organize(payload:OrganizeRequest):
        began=perf_counter(); status='OK'; error=None
        try:
            result=app.state.assistant.organize(payload)
            status='SAFETY_HANDOFF_REQUIRED' if result.safety.level!='NORMAL' else 'CLARIFICATION' if result.analysis.needsClarification else 'OK'
            return result
        except AssistantError as exc:
            status='ERROR'; error=exc.code; raise
        except Exception:
            status='ERROR'; error='INTERNAL_ERROR'
            # Never include exception text: it can contain database payloads or patient text.
            raise AssistantError('INTERNAL_ERROR','The assistant could not complete this request',500)
        finally:
            logger.info('requestId=%s module=ADHD_EXECUTIVE_FUNCTION operation=organize language=%s latencyMs=%.2f modelVersion=journal-linear-seed42-v1 resultStatus=%s errorCode=%s',payload.requestId,payload.patient.language,(perf_counter()-began)*1000,status,error)
    return app

app=create_app()
