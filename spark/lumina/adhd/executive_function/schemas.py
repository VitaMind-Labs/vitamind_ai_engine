from datetime import date, time
from typing import Literal, Annotated
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, AwareDatetime, model_validator

Language=Literal['EN','AR']
Capacity=Literal['HIGH','NORMAL','REDUCED','VERY_LOW']
Intent=Literal['ADD_TASK','ORGANIZE_DAY','PRIORITIZE','BREAK_DOWN_TASK','START_TASK','CONTINUE_TASK','FOCUS_HELP','INTERRUPTION_RECOVERY','TIME_ESTIMATION','OVERWHELMED_WITH_TASKS','PROCRASTINATION','TASK_COMPLETED','TASK_FAILED','RESCHEDULE','UNKNOWN']
Friction=Literal['TOO_BIG','UNCLEAR','BORING','ANXIETY','PERFECTIONISM','LOW_ENERGY','NO_CLEAR_START','TOO_MANY_CHOICES','TIME_BLINDNESS','DISTRACTION','INTERRUPTION','AVOIDANCE','OVERWHELM','UNKNOWN']
OutcomeValue=Literal['DONE','PARTIAL','NOT_STARTED','HELPFUL','NOT_HELPFUL','TOO_HARD','TOO_LONG','TOO_EASY','INTERRUPTED']
Text=Annotated[str,Field(min_length=1,max_length=500)]
Minutes=Annotated[StrictInt,Field(ge=1,le=1440)]

class Schema(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True,validate_assignment=True)

class Preferences(Schema):
    responseLength: Literal['SHORT','NORMAL']='SHORT'
    preferredFocusMinutes: Literal[5,10,15,25]|None=None
    planningStyle: str|None=None
    reminderStyle: str|None=None

class Patient(Schema):
    id: UUID
    language: Language
    supportTrack: Literal['ADHD']='ADHD'
    preferences: Preferences=Field(default_factory=Preferences)

class Message(Schema):
    text: Annotated[str,Field(min_length=1,max_length=4000)]

class CurrentState(Schema):
    energy: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    stress: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    capacity: Capacity|None=None
    location: str|None=None

class Task(Schema):
    temporaryId: Annotated[str,Field(min_length=1,max_length=80)]|None=None
    title: Text
    scheduledDate: date|None=None
    startTime: time|None=None
    deadline: date|None=None
    durationMinutes: Minutes|None=None
    importance: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    consequence: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    energyRequirement: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    startupDifficulty: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    cognitiveLoad: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    dependencies: list[Annotated[str,Field(min_length=1,max_length=80)]]=Field(default_factory=list,max_length=20)
    postponedCount: Annotated[StrictInt,Field(ge=0,le=1000)]=0
    location: str|None=None
    status: Literal['TODO','DONE','DEFERRED']='TODO'
    source: Literal['PROVIDED','MESSAGE','CALENDAR']='PROVIDED'

class TimeContext(Schema):
    referenceDate: date|None=None
    localTime: time|None=None
    availableMinutes: Annotated[StrictInt,Field(ge=0,le=1440)]|None=None
    day: date|None=None

class JournalSignals(Schema):
    sleep: Literal['LOW','NORMAL','UNKNOWN']='UNKNOWN'
    energy: Literal['LOW','NORMAL','HIGH','UNKNOWN']='UNKNOWN'
    stress: Literal['LOW','NORMAL','HIGH','UNKNOWN']='UNKNOWN'
    overwhelm: StrictBool=False
    safetyLevel: Literal['NORMAL','ELEVATED','CRISIS']='NORMAL'

class JournalContext(Schema):
    date: date
    signals: JournalSignals

class Memory(Schema):
    key: Literal['preferred_focus_duration','evening_complex_tasks','response_length']
    value: StrictInt|str
    confidence: Annotated[float,Field(ge=0,le=1)]
    evidenceCount: Annotated[StrictInt,Field(ge=1)]=1
    observedAt: AwareDatetime|None=None

class RecoveryContext(Schema):
    taskId: str
    lastCompletedStep: Text|None=None
    nextStep: Text|None=None
    timestamp: AwareDatetime

class Outcome(Schema):
    attemptId: UUID
    patientId: UUID
    timestamp: AwareDatetime
    outcome: OutcomeValue
    taskId: str|None=None
    focusMinutes: Literal[5,10,15,25]|None=None
    complexity: Literal['HIGH','NORMAL','LOW','UNKNOWN']='UNKNOWN'
    localHour: Annotated[StrictInt,Field(ge=0,le=23)]|None=None

class CalendarOptions(Schema):
    commit: StrictBool=False

class OrganizeRequest(Schema):
    requestId: UUID
    patient: Patient
    message: Message
    currentState: CurrentState=Field(default_factory=CurrentState)
    tasks: list[Task]=Field(default_factory=list,max_length=100)
    goals: list[Text]=Field(default_factory=list,max_length=10)
    recentContext: list[RecoveryContext]=Field(default_factory=list,max_length=10)
    relevantMemory: list[Memory]=Field(default_factory=list,max_length=30)
    journalContext: list[JournalContext]=Field(default_factory=list,max_length=30)
    outcomes: list[Outcome]=Field(default_factory=list,max_length=300)
    timeContext: TimeContext=Field(default_factory=TimeContext)
    calendar: CalendarOptions=Field(default_factory=CalendarOptions)

    @model_validator(mode='after')
    def ownership(self):
        if any(o.patientId!=self.patient.id for o in self.outcomes):
            raise ValueError('outcome patientId must match the request patient')
        ids=[t.temporaryId for t in self.tasks if t.temporaryId]
        if len(ids)!=len(set(ids)):
            raise ValueError('duplicate task IDs')
        return self

class PriorityFactor(Schema):
    name: str
    points: int
    evidence: str

class RankedTask(Schema):
    task: Task
    score: int
    factors: list[PriorityFactor]
    blocked: StrictBool=False
    bucket: Literal['PRIMARY','SECONDARY','OPTIONAL','DEFER']='OPTIONAL'

class NextAction(Schema):
    text: str
    estimatedMinutes: Minutes|None=None
    source: Literal['TEMPLATE','PROVIDED_CONTEXT']='TEMPLATE'
    templateId: str

class Plan(Schema):
    strategy: Literal['ONE_NEXT_ACTION','SHORT_DAY_PLAN','CLARIFY','NO_TIME','NO_TASKS','SAFETY_HANDOFF','TASK_COMPLETED_ACK']
    primaryTask: Task|None=None
    nextAction: NextAction|None=None
    secondaryTasks: list[Task]=Field(default_factory=list,max_length=2)
    rankedTasks: list[RankedTask]=Field(default_factory=list)
    day: date|None=None

class FocusSession(Schema):
    recommended: StrictBool=True
    minutes: Literal[5,10,15,25]
    successCondition: str
    basis: str
    estimateOfTaskDuration: StrictBool=False

class Candidate(Schema):
    key: str
    value: StrictInt|str
    category: Literal['PREFERENCE','PATTERN']
    confidence: Annotated[float,Field(ge=0,le=1)]
    evidenceCount: StrictInt
    evidenceDays: StrictInt
    explanation: str

class Analysis(Schema):
    taskCount: StrictInt
    needsClarification: StrictBool
    dataQuality: Literal['LIMITED','MODERATE','GOOD']
    uncertainty: list[str]
    clarification: str|None=None
    capacityEvidence: list[str]=Field(default_factory=list)
    intentSource: str='RULES'
    intentConfidence: float|None=None
    frictionSource: str='RULES'
    languagesObserved: list[str]=Field(default_factory=list)

class Response(Schema):
    text: str
    tone: Literal['CALM']='CALM'
    length: Literal['SHORT','NORMAL']='SHORT'
    templateId: str

class TaskOperation(Schema):
    operation: Literal['ADD','COMPLETE','RESCHEDULE','RECORD_OUTCOME']
    task: Task|None=None
    taskId: str|None=None
    status: Literal['PROPOSED','APPLIED']='PROPOSED'

class FollowUp(Schema):
    type: Literal['AFTER_ACTION','CLARIFICATION','NONE']
    recommendedAfterMinutes: Minutes|None=None

class Safety(Schema):
    level: Literal['NORMAL','ELEVATED','CRISIS']
    flags: list[str]=Field(default_factory=list)
    source: str
    deliveryStatus: Literal['NOT_SENT']='NOT_SENT'
    inputRemainsEditable: Literal[True]=True

class Performance(Schema):
    processingTimeMs: float

class OrganizeResponse(Schema):
    requestId: UUID
    module: Literal['ADHD_EXECUTIVE_FUNCTION']='ADHD_EXECUTIVE_FUNCTION'
    version: str='1.1.0'
    language: Language
    intent: Intent
    friction: list[Friction]
    capacity: Capacity
    analysis: Analysis
    plan: Plan
    focusSession: FocusSession|None=None
    response: Response
    taskOperations: list[TaskOperation]=Field(default_factory=list)
    memoryCandidates: list[Candidate]=Field(default_factory=list)
    patternCandidates: list[Candidate]=Field(default_factory=list)
    followUp: FollowUp
    safety: Safety
    performance: Performance
    tokenUsage: None=None
