"""Metadata-only summaries for clinician review; not medical recommendations."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from .schema import TIERS, RANK
from .signature import timestamp

def weekly_report(analyses, *, now=None, lang="en"):
    if lang not in ("en","ar"):
        raise ValueError("language must be en or ar")
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    # Last analysis of an entry wins; pauses must not inflate chart/alert counts.
    latest={}
    observed={"current":{},"previous":{}}
    for row in analyses:
        entry_id=row.get("entry_id")
        if not entry_id or row.get("tier") not in TIERS:
            raise ValueError("report requires valid analyses with stable entry_id")
        when=timestamp(row["analyzed_at"])
        if when>now or when<now-timedelta(days=14):
            continue
        period="current" if when>=now-timedelta(days=7) else "previous"
        prior=observed[period].get(entry_id,"none")
        observed[period][entry_id]=max((prior,row["tier"]),key=lambda tier:RANK[tier])
        if entry_id not in latest or when>timestamp(latest[entry_id]["analyzed_at"]):
            latest[entry_id]=row
    current=[r for r in latest.values() if timestamp(r["analyzed_at"])>=now-timedelta(days=7)]
    previous=[r for r in latest.values() if timestamp(r["analyzed_at"])<now-timedelta(days=7)]
    counts={tier:sum(r["tier"]==tier for r in current) for tier in TIERS}
    previous_counts={tier:sum(r["tier"]==tier for r in previous) for tier in TIERS}
    highest_counts={tier:sum(t==tier for t in observed["current"].values()) for tier in TIERS}
    highest_previous={tier:sum(t==tier for t in observed["previous"].values()) for tier in TIERS}
    recurrence=sum(r.get("signature",{}).get("threshold_reached",False) for r in current)
    unavailable=sum(not r.get("model",{}).get("available",False) for r in current)
    review=highest_counts["moderate"]+highest_counts["moderate_flagged"]+highest_counts["high"]
    change=highest_counts["high"]-highest_previous["high"]
    if lang=="en":
        bullets=[f"{len(current)} entries have their latest analysis in the last 7 days; {len(previous)} in the preceding 7 days.",f"{highest_counts['high']} distinct entries received a high support tier this week (change: {change:+d}); earlier high classifications are retained even after edits.",f"{review} entries are suggested for human review; no clinician notification was sent by this AI.",f"Personal phrase recurrence reached its configured threshold in {recurrence} entries.",f"The trained model was unavailable for {unavailable} latest entry analyses. Missing journal days cannot be interpreted as stability."]
    else:
        bullets=[f"عدد الإدخالات التي كان آخر تحليل لها في آخر 7 أيام: {len(current)}، وفي الأيام السبعة السابقة: {len(previous)}.",f"عدد الإدخالات التي حصلت على مستوى دعم مرتفع هذا الأسبوع: {highest_counts['high']} (التغير: {change:+d}). نحتفظ بهذا التصنيف السابق حتى بعد التعديل.",f"عدد الإدخالات المقترحة للمراجعة البشرية: {review}. لم يرسل هذا النموذج أي إشعار للطبيب.",f"بلغ تكرار العبارات الشخصية الحد المحدد في {recurrence} إدخالات.",f"تعذر استخدام النموذج المدرّب في آخر تحليل لعدد {unavailable} من الإدخالات. غياب الكتابة لا يعني استقرار الحالة."]
    return {"schema_version":"1.0","language":lang,"generated_at":now.isoformat(),"period_days":7,"entry_count":len(current),"tier_counts":counts,"previous_period_counts":previous_counts,"highest_observed_counts":highest_counts,"previous_highest_observed_counts":highest_previous,"summary_bullets":bullets,"clinician_review_required":True,"clinical_claim":"Journal support classifications only; no diagnosis, relapse confirmation or treatment recommendation.","raw_text_included":False,"chart":[{"entry_id":r["entry_id"],"at":r["analyzed_at"],"tier":r["tier"],"highest_observed_tier":observed["current"][r["entry_id"]],"marker":{"none":"○","low":"△","moderate":"◇","moderate_flagged":"!","high":"!!"}[r["tier"]]} for r in sorted(current,key=lambda r:r["analyzed_at"])]}
