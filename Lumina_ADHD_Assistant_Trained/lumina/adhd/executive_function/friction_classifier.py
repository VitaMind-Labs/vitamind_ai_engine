import re
from .language import normalize

RULES={
 'TOO_BIG':r'too big|huge task|الخطو[هة] كبير[هة]|المهم[هة] كبير[هة]',
 'UNCLEAR':r'unclear|what.*mean|not sure what|غير واضح|مو واضح',
 'BORING':r'boring|bored|ممل|ملل',
 'ANXIETY':r'anxious|worried|nervous|قلق|خايف',
 'PERFECTIONISM':r'perfect|mistake|مثالي|الكمال|اخطاء',
 'LOW_ENERGY':r'exhausted|tired|low energy|slept badly|مرهق|تعبان|طاقتي.*منخفض|نمت.*(?:قليل|سيئ)',
 'NO_CLEAR_START':r"can't start|cannot start|where to start|ما قدرت ابدا|من وين ابدا|مش قادر ابدا",
 'TOO_MANY_CHOICES':r'too many choices|which one|خيارات كثير|اي وحده',
 'TIME_BLINDNESS':r'lost track of time|time flies|وقت.*ضاع|ما حسيت بالوقت',
 'DISTRACTION':r'distract|concentrate|مشتت|ما اقدر اركز',
 'INTERRUPTION':r'interrupted|forgot what i was doing|نسيت.*كنت|قاطعني',
 'AVOIDANCE':r'avoid|procrastinat|putting.*off|اتجنب|اسوف|تسويف',
 'OVERWHELM':r'overwhelm|too many|وايد شغلات|مهام كثير|مب عارف من وين',
}
def classify_friction(text):
    text=normalize(text)
    return [label for label,pattern in RULES.items() if re.search(pattern,text)] or ['UNKNOWN']

class FrictionClassifier:
    def __init__(self):
        from .classical_model import ClassicalModel
        self.model=ClassicalModel('friction')
    def predict(self,text):
        if self.model.supports(text):
            output=self.model.predict(text)
            labels=[label for label,value in zip(self.model.artifact['labels'],output) if value=='1']
            if len(labels)>1 and 'UNKNOWN' in labels: labels.remove('UNKNOWN')
            return labels or ['UNKNOWN'],'LOCAL_TRAINED_CLASSIFIER'
        return classify_friction(text),'RULES'
