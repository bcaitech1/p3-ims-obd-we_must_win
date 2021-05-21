import pandas as pd
import numpy as np
from scipy import stats

# sample_submisson.csv 열기
sample = pd.read_csv('sample.csv', index_col=None)
submission_1 = pd.read_csv('1.csv', index_col=None)
submission_2 = pd.read_csv('2.csv', index_col=None)
submission_3 = pd.read_csv('3.csv', index_col=None)
submission_4 = pd.read_csv('4.csv', index_col=None)
submission_5 = pd.read_csv('5.csv', index_col=None)
submissions = [submission_1,submission_2,submission_3,submission_4,submission_5]



sample["image_id"] = submission_1["image_id"]
pred_string = []
for i in range(len(submission_1)):
    pred=[]
    for sub in submissions:
        pred.append(list(map(int,sub["PredictionString"][i].split())))

    pred = np.array(pred)
    mode_pred = list(map(str, stats.mode(pred)[0][0]))  
    pred_string.append(' '.join(mode_pred))

sample["PredictionString"] = pred_string

    
    
sample.to_csv("submission_ensemble.csv", index=False)
print('submission saved!')

