"""Native-language model inputs. These authored phrases are not diagnoses."""
import re
from ..clinical.language import normalize, patient_clauses

INPUT_TYPE = 'patient_text_bilingual_v1'

# Parallel descriptions of the existing feature inventory. Kept outside the
# extractor: classifier suggestions must not silently create patient evidence.
FEATURE_TEXT = {
 'childhood_onset': {
  'en': ['These difficulties have been there since childhood', 'I remember these problems from primary school', 'This began when I was a child'],
  'ar': ['هذه الصعوبات موجودة عندي منذ الطفولة', 'أتذكر هذه المشاكل من أيام المدرسة الابتدائية', 'بدأت هذه الأمور عندما كنت صغيرا']},
 'distractibility': {
  'en': ['I get distracted and lose focus', 'My attention wanders while I am doing things', 'I struggle to stay focused on a task'],
  'ar': ['أنا مشتت وأفقد تركيزي بسهولة', 'أسرح كثيرا أثناء القيام بمهامي', 'يصعب علي الاستمرار في التركيز على مهمة']},
 'forgetfulness': {
  'en': ['I forget appointments and lose things', 'I keep misplacing my belongings and forgetting plans', 'I often forget what I was about to do'],
  'ar': ['أنسى المواعيد وأضيع أغراضي', 'تضيع مني حاجاتي وأنسى ما خططت له', 'كثيرا ما أنسى ما كنت أريد فعله']},
 'procrastination': {
  'en': ['I put off tasks until the last moment', 'I procrastinate even when I want to start', 'I keep delaying things I need to do'],
  'ar': ['أؤجل المهام حتى آخر لحظة', 'أسوف حتى عندما أريد البدء', 'أستمر في تأجيل ما يجب علي فعله']},
 'hyperactivity_impulsivity': {
  'en': ['I cannot sit still and act without thinking', 'I keep fidgeting and interrupting impulsively', 'I feel restless and make impulsive decisions'],
  'ar': ['لا أستطيع الجلوس بهدوء وأتصرف دون تفكير', 'أتململ كثيرا وأقاطع الآخرين باندفاع', 'أشعر بالتململ وأتخذ قرارات اندفاعية']},
 'reduced_sleep': {
  'en': ['I have been sleeping only three hours a night', 'I sleep much less than usual', 'I barely get any sleep at night'],
  'ar': ['أنام ثلاث ساعات فقط في الليل', 'أنام أقل بكثير من المعتاد', 'بالكاد أنام خلال الليل']},
 'decreased_need_for_sleep': {
  'en': ['Despite little sleep I still feel full of energy', 'I need very little sleep and do not feel tired', 'I stay energetic even after sleeping so little'],
  'ar': ['رغم قلة النوم لا أشعر بالتعب وما زلت مليئا بالطاقة', 'أحتاج إلى نوم قليل جدا دون أن أشعر بالإرهاق', 'أبقى نشيطا رغم أنني نمت ساعات قليلة']},
 'elevated_mood': {
  'en': ['My mood is unusually high and I feel energized', 'I feel much more excited and energetic than usual', 'I have been feeling on top of the world'],
  'ar': ['مزاجي مرتفع بشكل غير معتاد وأشعر بطاقة كبيرة', 'أشعر بحماس ونشاط أكثر بكثير من المعتاد', 'أشعر بأنني في قمة السعادة بصورة غير مألوفة']},
 'grandiosity': {
  'en': ['I feel invincible and believe I can do anything', 'I feel I have exceptional abilities far beyond others', 'I think I am the smartest person and cannot fail'],
  'ar': ['أشعر أنني لا أقهر وأنني أستطيع فعل أي شيء', 'أشعر أن لدي قدرات استثنائية تفوق الآخرين كثيرا', 'أظن أنني أذكى شخص وأن الفشل مستحيل بالنسبة لي']},
 'racing_thoughts': {
  'en': ['My thoughts are racing', 'My mind is moving too fast to keep up with', 'Ideas rush through my head without stopping'],
  'ar': ['أفكاري تتسارع كثيرا', 'عقلي يتحرك بسرعة لا أستطيع مجاراتها', 'تتدفق الأفكار في رأسي دون توقف']},
 'pressured_speech': {
  'en': ['I talk very fast and cannot stop talking', 'People say my speech is unusually rapid', 'I keep talking nonstop and others cannot interrupt'],
  'ar': ['أتكلم بسرعة شديدة ولا أستطيع التوقف عن الكلام', 'يقول الناس إن كلامي سريع بشكل غير معتاد', 'أستمر في الحديث دون توقف ولا يستطيع الآخرون مقاطعتي']},
 'flight_of_ideas': {
  'en': ['I jump from one topic to another', 'I keep moving from one idea to the next', 'My conversation rapidly switches between unrelated topics'],
  'ar': ['أقفز من موضوع إلى آخر', 'أنتقل باستمرار من فكرة إلى فكرة أخرى', 'يتنقل كلامي بسرعة بين مواضيع غير مترابطة']},
 'impulsive_spending': {
  'en': ['I spent far too much money on things I did not need', 'I went on an unusual spending spree', 'I bought lots of things impulsively and overspent'],
  'ar': ['أنفقت مالا كثيرا على أشياء لا أحتاجها', 'مررت بنوبة إنفاق غير معتادة', 'اشتريت أشياء كثيرة باندفاع وتجاوزت ميزانيتي']},
 'increased_goal_directed_activity': {
  'en': ['I started many projects at the same time', 'I suddenly took on much more work than usual', 'I am working on numerous new plans all at once'],
  'ar': ['بدأت مشاريع كثيرة في الوقت نفسه', 'تحملت فجأة أعمالا أكثر بكثير من المعتاد', 'أعمل على خطط جديدة عديدة دفعة واحدة']},
 'episodic_pattern': {
  'en': ['This happens in distinct periods that come and go', 'These changes occur in episodes separated by calmer periods', 'I have repeated stretches like this with returns to my usual self'],
  'ar': ['يحدث هذا في فترات واضحة تأتي وتذهب', 'تظهر هذه التغيرات على شكل نوبات تفصل بينها فترات أهدأ', 'تتكرر فترات كهذه ثم أعود إلى حالتي المعتادة']},
 'depression_alternation': {
  'en': ['After these periods I crash into deep depression', 'I alternate between these periods and feeling very low', 'Then I feel empty and struggle to get out of bed'],
  'ar': ['بعد هذه الفترات أنهار في اكتئاب عميق', 'أتناوب بين هذه الفترات وبين الشعور بانخفاض شديد في المزاج', 'ثم أشعر بالفراغ وأجد صعوبة في النهوض من السرير']},
 'auditory_perceptual_experience': {
  'en': ['I hear voices when nobody is there', 'I hear people speaking although I am alone', 'I hear sounds or voices that others do not hear'],
  'ar': ['أسمع أصواتا عندما لا يوجد أحد', 'أسمع أشخاصا يتكلمون رغم أنني وحدي', 'أسمع أصواتا لا يسمعها الآخرون']},
 'persecutory_ideas': {
  'en': ['I feel that people are watching me', 'I believe someone is following me and plotting against me', 'I feel targeted and watched even when others disagree'],
  'ar': ['أشعر أن الناس يراقبونني', 'أعتقد أن شخصا يتبعني ويتآمر ضدي', 'أشعر أنني مستهدف ومراقب حتى عندما يختلف الآخرون معي']},
 'thought_disorganization': {
  'en': ['My thoughts feel jumbled and hard to organize', 'I cannot put my thoughts in order', 'I find it difficult to connect my thoughts coherently'],
  'ar': ['أفكاري مشوشة ويصعب ترتيبها', 'لا أستطيع تنظيم أفكاري', 'أجد صعوبة في ربط أفكاري بشكل واضح']},
 'social_withdrawal': {
  'en': ['I stopped seeing my friends and isolate myself', 'I withdraw from people and stay in my room', 'I avoid social contact and spend most of my time alone'],
  'ar': ['توقفت عن رؤية أصدقائي وأعزل نفسي', 'أنسحب من الناس وأبقى في غرفتي', 'أتجنب التواصل مع الآخرين وأقضي معظم وقتي وحدي']},
}


def text_key(text):
    """Deduplicate after the same normalization used by the vectorizers."""
    return ' '.join(re.sub(r'[^\w\s]', ' ', normalize(text)).split())


def build_input(state, text=None):
    language = 'ar' if state.context.language == 'ar' else 'en'
    from ..clinical.feature_extractor import RULES
    used = [r.feature for r in RULES if state.get_status(r.domain, r.feature).value == 'present']
    parts = []
    if text and len(normalize(text).split()) >= 3:
        parts.extend(clause for clause, own in patient_clauses(text) if own)
    # Native-language evidence anchors carry the meaning of contextual yes/no
    # replies without sending the assistant question or a guessed diagnosis.
    parts.extend(FEATURE_TEXT[feature][language][0] for feature in used)
    return '. '.join(dict.fromkeys(parts)), used
