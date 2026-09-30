"""Small categorical metrics for the development report; no sklearn at runtime."""
def accuracy_score(expected,predicted):
    return sum(a==b for a,b in zip(expected,predicted))/len(expected) if expected else 0.0

def classification_report(expected,predicted,output_dict=True,zero_division=0):
    result={}
    for label in sorted(set(expected)|set(predicted)):
        tp=sum(a==label and b==label for a,b in zip(expected,predicted))
        support=expected.count(label); predicted_count=predicted.count(label)
        precision=tp/predicted_count if predicted_count else zero_division
        recall=tp/support if support else zero_division
        f1=2*precision*recall/(precision+recall) if precision+recall else zero_division
        result[label]={'precision':precision,'recall':recall,'f1-score':f1,'support':support}
    classes=list(result.values()); total=len(expected)
    result['accuracy']=accuracy_score(expected,predicted)
    result['macro avg']={key:sum(row[key] for row in classes)/len(classes) if classes else 0 for key in ('precision','recall','f1-score')}
    result['weighted avg']={key:sum(row[key]*row['support'] for row in classes)/total if total else 0 for key in ('precision','recall','f1-score')}
    result['macro avg']['support']=result['weighted avg']['support']=total
    return result

def f1_score(expected,predicted,average='macro',zero_division=0):
    if average!='macro': raise ValueError('This development helper supports macro F1 only.')
    return classification_report(expected,predicted,zero_division=zero_division)['macro avg']['f1-score']
