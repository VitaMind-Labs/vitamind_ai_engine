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
    else:
        text=choose(lang,f'Write one small first step for: {title}.',f'اكتب خطوة اولى صغيرة للمهمة: {title}.'); key='generic.define_step'
    return NextAction(text=text,templateId=key,estimatedMinutes=None)
