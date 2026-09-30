import re
from .language import normalize,choose
from .schemas import NextAction

def first_action(task,lang,capacity):
    title=task.title
    s=normalize(title)
    if re.search(r'presentation|slides|عرض|شرائح',s):
        text=choose(lang,'Open the presentation file.','افتح ملف العرض التقديمي.'); key='presentation.open'
    elif re.search(r'email|e-mail|ارسل|بريد',s):
        text=choose(lang,f'Open a draft for: {title}.',f'افتح مسودة رسالة للمهمة: {title}.'); key='message.draft'
    elif re.search(r'call|اكلم|اتصل',s):
        text=choose(lang,f'Find the contact details for: {title}.',f'ابحث عن بيانات الاتصال للمهمة: {title}.'); key='call.contact'
    elif re.search(r'report|assignment|document|تقرير|واجب|مستند',s):
        text=choose(lang,f'Open the document for: {title}.',f'افتح الملف الخاص بالمهمة: {title}.'); key='document.open'
    elif re.search(r'buy|grocer|اشتري|اشتر',s):
        text=choose(lang,f'Write the first item needed for: {title}.',f'اكتب اول غرض تحتاجه للمهمة: {title}.'); key='shopping.first_item'
    elif re.search(r'clean|wash|انظف|اغسل',s):
        text=choose(lang,f'Put one item in place to start: {title}.',f'ضع غرضا واحدا في مكانه لتبدا: {title}.'); key='clean.one_item'
    elif re.search(r'study|read|review|ادرس|اذاكر|اقرا|اراجع',s):
        text=choose(lang,f'Open the material for: {title}.',f'افتح المادة الخاصة بالمهمة: {title}.'); key='study.open'
    elif re.search(r'\b(?:doctor|dentist|clinic|hospital|appointment|therapist|psychiatrist|checkup)\b|طبيب|دكتور|عياد|موعد|مستشفى',s):
        text=choose(lang,'Check the time and place, and set a reminder to leave.','تاكد من الوقت والمكان، واضبط تذكيرا للخروج.'); key='appointment.prepare'
    elif re.search(r'\b(?:pray|prayer|salah|salat)\b|صلا[هة]|اصلي|صلي',s):
        text=choose(lang,'Pause what you are doing and get ready for prayer.','توقف عما تفعله وتهيأ للصلاة.'); key='prayer.prepare'
    elif re.search(r'\b(?:see|meet|visit|catch up with|hang out with)\b.*\b(?:friend|friends|mom|mum|dad|family|sister|brother)\b|اشوف|اقابل|ازور|صديق|اصدقاء',s):
        text=choose(lang,'Send a short message to fix a time.','ارسل رسالة قصيرة لتحديد موعد.'); key='social.arrange'
    elif re.search(r'workout|work out|gym|training|تمرين|نادي',s):
        text=choose(lang,'Put on your workout clothes and fill your water bottle.','البس ملابس التمرين وجهز زجاجة الماء.'); key='workout.prepare'
    elif re.search(r'school|class|lecture|homework|مدرس|جامع|محاضر',s):
        text=choose(lang,'Pack your bag and check the time you need to leave.','جهز حقيبتك وتاكد من وقت الخروج.'); key='school.prepare'
    else:
        text=choose(lang,f'Write one small first step for: {title}.',f'اكتب خطوة اولى صغيرة للمهمة: {title}.'); key='generic.define_step'
    return NextAction(text=text,templateId=key,estimatedMinutes=None)
