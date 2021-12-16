#%% read the data and define the types
end="OS" #end="OS" or "DFS"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import copy
from pandas.api.types import CategoricalDtype

df=pd.read_excel(r'C:\code\df.xlsx')

#deefine type of features
df["os"]=df["os"].astype("category")
df["sex"]=df["sex"].astype("category")
df["menopause"]=df["menopause"].astype("category")
df["side"]=df["side"].astype("category")
df["invasive"]=df["invasive"].astype("category")
df["TNM"]=df["TNM"].astype("category")
df["multi"]=df["multi"].astype("category")
df["Ki_67"]=df["Ki_67"].astype("category")
df["ER"]=df["ER"].astype("category")
df["PR"]=df["PR"].astype("category")
df["HER2"]=df["HER2"].astype("category")
df["breast_surgery"]=df["breast_surgery"].astype("category")
df["axillary_surgery"]=df["axillary_surgery"].astype("category")
df["rebuild_surgery"]=df["rebuild_surgery"].astype("category")
df["adjuvant_chemotherapy"]=df["adjuvant_chemotherapy"].astype("category")
df["targeted_therapy"]=df["targeted_therapy"].astype("category")
df["adjuvant_radiotherapy"]=df["adjuvant_radiotherapy"].astype("category")
df["adjuvant_endocrine_therapy"]=df["adjuvant_endocrine_therapy"].astype("category")
df["neoadjuvant"]=df["neoadjuvant"].astype("category")

#%% delete observations with too many missing values
#miss_r is the results of logrank test

miss_r = pd.DataFrame(columns = ["No. of Missing fields","Excluded patients", 
                                 "Included patients", 
                                 "Log-rank statistics", 
                                 "Log-rank p-value"]) 
miss_r2 = pd.DataFrame(columns = ["No. of Missing fields","Excluded patients", 
                                 "Included patients", 
                                 "Log-rank statistics", 
                                 "Log-rank p-value"]) 

print(df.dtypes)

from sksurv.datasets import get_x_y
from sklearn.model_selection import train_test_split
X, y = get_x_y(df,attr_labels=['os', 'os_mon'], pos_label=1)
X0=copy.deepcopy(X)
y0=copy.deepcopy(y)

rows_null = df.shape[1] - df.count(axis=1) #count missing values in each row
df['rows_null']=rows_null
misscount=df.groupby('rows_null').size()
from sksurv.compare import compare_survival
for i in range(0,max(rows_null)):     #1-3
    df['in']=pd.cut(df.rows_null,[0, i+0.1, max(rows_null) ], 
                    right=True, 
                    labels=[1,0],
                    include_lowest=True)
    df_Old=copy.deepcopy(df)
    df_Old['log']='all'
    df_In=df[df['in'].isin([1])]
    df_In['log']='in'
    df_Old=df_Old.append(df_In)
    Xlog, ylog = get_x_y(df_Old,attr_labels=['os', 'os_mon'], pos_label=1)
    logrank = compare_survival(ylog, df_Old['log'], return_stats=True)
    miss_r.loc[i] = 0
    miss_r.iloc[i,0]=i+1
    miss_r.iloc[i,1]=logrank[2].iloc[0,0]-logrank[2].iloc[1,0]
    miss_r.iloc[i,2]=logrank[2].iloc[1,0]
    miss_r.iloc[i,3]=logrank[0]
    miss_r.iloc[i,4]=logrank[1]
    print(i)

for i in range(0,max(rows_null)):    #1-3
    df['in']=pd.cut(df.rows_null,[0, i+0.1, max(rows_null) ], 
                    right=True, 
                    labels=[1,0],
                    include_lowest=True)
    logrank = compare_survival(y, df['in'], return_stats=True)
    miss_r2.loc[i] = 0
    miss_r2.iloc[i,0]=i+1
    miss_r2.iloc[i,1]=logrank[2].iloc[0,0]
    miss_r2.iloc[i,2]=logrank[2].iloc[1,0]
    miss_r2.iloc[i,3]=logrank[0]
    miss_r2.iloc[i,4]=logrank[1]
    print(i)

missd = r"C:\code\miss" +end+ ".csv"
miss_r.to_csv(missd,index=False,header=True)

#delete observations missing x or more values, the x in our study is 3 
df=df[df['rows_null']<3]
col_null_after= df.isnull().sum(axis=0)
df=df.drop(["sex"],axis=1)
df=df.drop(["rows_null"],axis=1)
df=df.drop(["id"],axis=1)
df=df.drop(["in"],axis=1)
df.rename(columns={'os':'OS','os_mon':'OS_Mon'},inplace=True)

#%% use missforest to fill in missing value
df["OS"]=df["OS"].astype("category")
df["menopause"]=df["menopause"].astype("category")
df["side"]=df["side"].astype("category")
df["invasive"]=df["invasive"].astype("category")
df["TNM"]=df["TNM"].astype("category")
df["multi"]=df["multi"].astype("category")
df["Ki_67"]=df["Ki_67"].astype("category")
df["ER"]=df["ER"].astype("category")
df["PR"]=df["PR"].astype("category")
df["HER2"]=df["HER2"].astype("category")
df["breast_surgery"]=df["breast_surgery"].astype("category")
df["axillary_surgery"]=df["axillary_surgery"].astype("category")
df["rebuild_surgery"]=df["rebuild_surgery"].astype("category")
df["adjuvant_chemotherapy"]=df["adjuvant_chemotherapy"].astype("category")
df["targeted_therapy"]=df["targeted_therapy"].astype("category")
df["adjuvant_radiotherapy"]=df["adjuvant_radiotherapy"].astype("category")
df["adjuvant_endocrine_therapy"]=df["adjuvant_endocrine_therapy"].astype("category")
df["neoadjuvant"]=df["neoadjuvant"].astype("category")

dfone = OneHotEncoder().fit_transform(df)

#Get rid of the ".0" in the variable name
col=list(dfone.columns)
for i in range(len(col)):
    col[i]=col[i].replace('.0','')
    print (i)
dfone.columns = col
dfone.to_csv(r"C:\code\dfone.csv",index=True,header=True)

#calculate the time of process cause it will take a long time if the dataset is large
import time
t0=time.time()
print('Display the start time of the program:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

#use missforest to fill in missing value
from missingpy import MissForest
imputer = MissForest(random_state=1)
dfonecopy=copy.deepcopy(dfone)
dfone=dfone.drop(["OS=1","OS_Mon"],axis=1)
dft = imputer.fit_transform(dfone,cat_vars=np.array([dfone.columns.get_loc('menopause=1'),
                                                   dfone.columns.get_loc('side=1'),
                                                   dfone.columns.get_loc('invasive=1'),
                                                   dfone.columns.get_loc('multi=1'),
                                                   dfone.columns.get_loc('TNM=2'),
                                                   dfone.columns.get_loc('TNM=3'),
                                                   dfone.columns.get_loc('TNM=4'),
                                                   dfone.columns.get_loc('Ki_67=1'),
                                                   dfone.columns.get_loc('ER=1'),
                                                   dfone.columns.get_loc('PR=1'),
                                                   dfone.columns.get_loc('HER2=1'),
                                                   dfone.columns.get_loc('breast_surgery=1'),
                                                   dfone.columns.get_loc('breast_surgery=2'),
                                                   dfone.columns.get_loc('axillary_surgery=1'),
                                                   dfone.columns.get_loc('axillary_surgery=2'),
                                                   dfone.columns.get_loc('axillary_surgery=3'),
                                                   dfone.columns.get_loc('rebuild_surgery=1'),
                                                   dfone.columns.get_loc('rebuild_surgery=2'),
                                                   dfone.columns.get_loc('adjuvant_chemotherapy=1'),
                                                   dfone.columns.get_loc('targeted_therapy=1'),
                                                   dfone.columns.get_loc('adjuvant_radiotherapy=1'),
                                                   dfone.columns.get_loc('adjuvant_endocrine_therapy=1'),
                                                   dfone.columns.get_loc('neoadjuvant=1')                                                    
                                                   ]))

t1=time.time()
print('Display the program end time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("Total time：%.6fs"%(t1-t0))

dft=pd.DataFrame(dft)
dft.columns = list(dfone)
dfonecopy = dfonecopy.reset_index(drop=True)
dft = pd.concat([dfonecopy.iloc[:,0:2],dft],axis=1)
dft.to_csv(r"C:\code\dft.csv",index=False,header=True)

df=dft
df["OS=1"]=df["OS=1"].astype("category")
df["menopause=1"]=df["menopause=1"].astype("category")
df["side=1"]=df["side=1"].astype("category")
df["invasive=1"]=df["invasive=1"].astype("category")
df["TNM=2"]=df["TNM=2"].astype("category")
df["TNM=3"]=df["TNM=3"].astype("category")
df["TNM=4"]=df["TNM=4"].astype("category")
df["multi=1"]=df["multi=1"].astype("category")
df["Ki_67=1"]=df["Ki_67=1"].astype("category")
df["ER=1"]=df["ER=1"].astype("category")
df["PR=1"]=df["PR=1"].astype("category")
df["HER2=1"]=df["HER2=1"].astype("category")
df["breast_surgery=1"]=df["breast_surgery=1"].astype("category")
df["axillary_surgery=1"]=df["axillary_surgery=1"].astype("category")
df["rebuild_surgery=1"]=df["rebuild_surgery=1"].astype("category")
df["adjuvant_chemotherapy=1"]=df["adjuvant_chemotherapy=1"].astype("category")
df["targeted_therapy=1"]=df["targeted_therapy=1"].astype("category")
df["adjuvant_radiotherapy=1"]=df["adjuvant_radiotherapy=1"].astype("category")
df["adjuvant_endocrine_therapy=1"]=df["adjuvant_endocrine_therapy=1"].astype("category")
df["neoadjuvant=1"]=df["neoadjuvant=1"].astype("category")
print(dft.dtypes)

df.rename(columns={'OS=1':'OS'},inplace=True)
if end == "DFS":
    df=df[~df['TNM=4'].isin([1])]
    df=df.drop(["TNM=4"],axis=1)
    
dfcopy=copy.deepcopy(df)
#%% Split training set and test set
df=copy.deepcopy(dfcopy)

from sksurv.datasets import get_x_y
from sklearn.model_selection import train_test_split
X, y = get_x_y(df,attr_labels=['OS', 'OS_Mon'], pos_label=1)
X0=X
y0=y
X, Xtest, y, ytest = train_test_split(X, y, test_size=0.3, random_state=1)
Xt=X
Xttest=Xtest
#%% We use R to compare C-index of different models and generate ROC curves, so we output the predict score
def outroc(ytest,preds,name):
    rocdata= pd.DataFrame() 
    rocdata['os']=ytest['OS']
    rocdata['time']=ytest['OS_Mon']
    rocdata['marker']=preds
    rocdata=rocdata.replace(True,1)
    rocdata=rocdata.replace(False,0)
    path=r"C:\code\rocdata"+name+".csv"
    rocdata.to_csv(path,index=False,header=True)


#%% COX
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cph2 = CoxPHSurvivalAnalysis()
#Variables that were not significant in univariate analysis were excluded
Xt2=Xt.drop(["side=1"],axis=1)
Xt2=Xt2.drop(["multi=1"],axis=1)
Xttest2=Xttest.drop(["side=1"],axis=1)
Xttest2=Xttest2.drop(["multi=1"],axis=1)
dfcox=df.drop(["side=1","multi=1"],axis=1)

#Variables with P > 0.05 in multivariate Cox regression were excluded
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
from lifelines.utils import median_survival_times
from lifelines import CoxPHFitter

cph3 = CoxPHFitter(penalizer = 0.1)
cph3.fit(dfcox, 'OS_Mon', 'OS')
cph3.print_summary(columns=["coef", "se(coef)", "p"], decimals=3)
cph3r=cph3.summary
cph3rvar=cph3r[cph3r["p"]>0.05].index #cph3rvar is a list of index of the variables with a P>0.05

Xt2=Xt2.drop(cph3rvar,axis=1)
Xttest2=Xttest2.drop(cph3rvar,axis=1)
Xt2.columns

cph2.fit(Xt2, y)

ytestlist=ytest.tolist()
ytestlist=[i[1] for i in ytestlist]
va_times = np.arange(min(ytestlist), max(ytestlist), 2)

# estimate performance on training data, thus use `va_y` twice.
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc,
                            brier_score)

va_auc0, va_mean_auc0 = cumulative_dynamic_auc(y, ytest, cph2.predict(Xttest2), va_times)
outroc(ytest, cph2.predict(Xttest2),"cox2")
#obtain the fig of time-dependent AUC at different time
plt.figure()
plt.plot(va_times, va_auc0, marker="o",markersize=3.,label ='cox2')
plt.axhline(va_mean_auc0, linestyle="--")
plt.xlabel("months from enrollment")
plt.ylabel("time-dependent AUC")
plt.legend()
plt.grid(True)

cindex = concordance_index_censored(ytest["OS"], ytest["OS_Mon"], cph2.predict(Xttest2))
print(cindex)
cox2cindex=cindex

est2=cph2.fit(Xt2, y)
survs = est2.predict_survival_function(Xttest2)

preds = [fn(36) for fn in survs]
times, cox2score3y = brier_score(y, ytest, preds, 36)
preds = [fn(60) for fn in survs]
times, cox2score5y = brier_score(y, ytest, preds, 60)
preds = [fn(120) for fn in survs]
times, cox2score10y = brier_score(y, ytest, preds, 120)



#%% COX-EN

#how the coefficients change for varying α
alphas = 10. ** np.linspace(-4, 4, 30)
coefficients = {}

cph = CoxPHSurvivalAnalysis()

for alpha in alphas:
    cph.set_params(alpha=alpha)
    cph.fit(Xt, y)
    key = round(alpha, 5)
    coefficients[key] = cph.coef_

coefficients = (pd.DataFrame
    .from_dict(coefficients)
    .rename_axis(index="feature", columns="alpha")
    .set_index(Xt.columns))

def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef, str(name).replace('=1','') + "   ",
            horizontalalignment="right",
            verticalalignment="center"
        )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)
cox_elastic_net.fit(Xt, y)

coefficients_elastic_net = pd.DataFrame(
    cox_elastic_net.coef_,
    index=Xt.columns,
    columns=np.round(cox_elastic_net.alphas_, 5)
)

plot_coefficients(coefficients_elastic_net, n_highlight=7)


#  use cross-validation to determine which subset and α generalizes best
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

coxnet_pipe = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100)
)
warnings.simplefilter("ignore", ConvergenceWarning)
coxnet_pipe.fit(Xt, y)

cv = KFold(n_splits=5, shuffle=True, random_state=1)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in alphas]},
    cv=cv,
    error_score=0.5,
    n_jobs=4).fit(Xt, y)

cv_results = pd.DataFrame(gcv.cv_results_)

alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)

#feature importance
best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs = pd.DataFrame(
    best_model.coef_,
    index=Xt.columns,
    columns=["coefficient"])

non_zero = np.sum(best_coefs.iloc[:,0] != 0)
print("Number of non-zero coefficients: {}".format(non_zero))

non_zero_coefs = best_coefs.query("coefficient != 0")

def strip_map(x):
    return x.strip('=1')

def tm(x):
    if x=="ln_transfer":
        return "ln_metastasis"
    else:
        return x

non_zero_coefs.index=non_zero_coefs.index.map(strip_map)
non_zero_coefs0=non_zero_coefs
non_zero_coefs.index=non_zero_coefs.index.map(tm)

coef_order = non_zero_coefs.abs().sort_values("coefficient").index

_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)

#fit the Cox-EN model

cph.set_params(alpha=gcv.best_params_["coxnetsurvivalanalysis__alphas"][0])
cph.fit(Xt, y)

ytestlist=ytest.tolist()
ytestlist=[i[1] for i in ytestlist]
 
va_times = np.arange(min(ytestlist), max(ytestlist), 2)

# estimate performance on training data, thus use `va_y` twice.
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)

va_auc, va_mean_auc = cumulative_dynamic_auc(y, ytest, cph.predict(Xttest), va_times)
plt.figure()
plt.plot(va_times, va_auc, marker="o",markersize=3.,label ='cox')
plt.axhline(va_mean_auc, linestyle="--")
plt.plot(va_times, va_auc0, marker="o",markersize=3.,label ='cox2')
plt.axhline(va_mean_auc0, linestyle="--")

plt.xlabel("months from enrollment")
plt.ylabel("time-dependent AUC")
plt.legend()
plt.grid(True)

cindex = concordance_index_censored(ytest["OS"], ytest["OS_Mon"], cph.predict(Xttest))
print(cindex)
coxcindex=cindex

from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score
from sksurv.preprocessing import OneHotEncoder

est=cph.fit(Xt, y)
survs = est.predict_survival_function(Xttest)
preds = [fn(36) for fn in survs]
times, coxscore3y = brier_score(y, ytest, preds, 36)

preds = [fn(60) for fn in survs]
times, coxscore5y = brier_score(y, ytest, preds, 60)

preds = [fn(120) for fn in survs]
times, coxscore10y = brier_score(y, ytest, preds, 120)

outroc(ytest,preds,"cox")

#%% SVM
Xt0=copy.deepcopy(Xt)
Xttest0=copy.deepcopy(Xttest)

Xt.columns=Xt.columns.map(strip_map)
Xttest.columns=Xttest.columns.map(strip_map)

Xt=Xt[non_zero_coefs.index.tolist()]
Xttest=Xttest[non_zero_coefs.index.tolist()]

print("coxcindex=",coxcindex)
print("coxscore3y=",coxscore3y)
print("coxscore5y=",coxscore5y)
print("coxscore10y=",coxscore10y)

X=copy.deepcopy(Xt)
y=copy.deepcopy(y)

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, GridSearchCV

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.column import encode_categorical
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM

sns.set_style("whitegrid")

#the amount of censoring for this data
n_censored = y.shape[0] - y["OS"].sum()
print("%.1f%% of records are censored" % (n_censored / y.shape[0] * 100))

plt.figure(figsize=(9, 6))
val, bins, patches = plt.hist((y["OS_Mon"][y["OS"]],
                               y["OS_Mon"][~y["OS"]]),
                              bins=30, stacked=True)
_ = plt.legend(patches, ["Time of Death", "Time of Censoring"])

estimator = FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=1)

#a function for evaluating the performance of models during grid search
def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['OS'], y['OS_Mon'], prediction)
    return result[0]

param_grid = {'alpha': 2. ** np.arange(-12, 13, 2)}
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
gcv = GridSearchCV(estimator, param_grid, scoring=score_survival_model,
                   n_jobs=4, iid=False, refit=False,
                   cv=cv)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
gcv = gcv.fit(Xt, y)

round(gcv.best_score_, 3), gcv.best_params_

def plot_performance(gcv):
    n_splits = gcv.cv.n_splits
    cv_scores = {"alpha": [], "test_score": [], "split": []}
    order = []
    for i, params in enumerate(gcv.cv_results_["params"]):
        name = "%.5f" % params["alpha"]
        order.append(name)
        for j in range(n_splits):
            vs = gcv.cv_results_["split%d_test_score" % j][i]
            cv_scores["alpha"].append(name)
            cv_scores["test_score"].append(vs)
            cv_scores["split"].append(j)
    df = pandas.DataFrame.from_dict(cv_scores)
    _, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(x="alpha", y="test_score", data=df, order=order, ax=ax)
    _, xtext = plt.xticks()
    for t in xtext:
        t.set_rotation("vertical")
        
plot_performance(gcv)

estimator.set_params(**gcv.best_params_)
estimator.fit(Xt, y)
estimator.predict(Xttest)
svmcindex=estimator.score(Xttest,ytest)

outroc(ytest,estimator.predict(Xttest),"svm")

# estimate performance on training data, thus use `va_y` twice.
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)

va_auc, va_mean_auc = cumulative_dynamic_auc(y, ytest, cph.predict(Xttest0), va_times)
va_auc2, va_mean_auc2 = cumulative_dynamic_auc(y, ytest, estimator.predict(Xttest), va_times)
plt.figure()
plt.plot(va_times, va_auc, marker="o")

plt.axhline(va_mean_auc, linestyle="--")

plt.plot(va_times, va_auc2, marker="o")
plt.axhline(va_mean_auc2, linestyle="--")

plt.xlabel("months from enrollment")
plt.ylabel("time-dependent AUC")
plt.grid(True)

#%% RSF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import RandomizedSearchCV

#Search for best hyper-parameter
random_state = 1

n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]
max_depth = [int(x) for x in np.linspace(5, 55, 11)]
max_features = ['auto', 'sqrt', 'log2']

min_samples_split = [int(x) for x in np.linspace(2, 10, 5)]
min_samples_leaf = [int(x) for x in np.linspace(2, 10, 5)]

rf=RandomSurvivalForest()

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

grid_search = RandomizedSearchCV(estimator=rf, 
                                  n_iter=50,
                               param_distributions=random_grid,
                               random_state=1
                               , n_jobs=4
                               )

import time
t0=time.time()
print('the start time of the program:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

grid_search.fit(Xt, y)

t1=time.time()
print('the end time of the program:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("total time：%.6fs"%(t1-t0))

rsf=RandomSurvivalForest(random_state=1)

rsf.set_params(**grid_search.best_params_)
est=rsf.fit(Xt, y)

#get the predictive performance of RSF
surv = rsf.predict_survival_function(Xttest, return_array=True)
surv2 = rsf.predict_cumulative_hazard_function(Xttest, return_array=True)
rsf.predict(Xttest)

est=rsf.fit(Xt, y)
surv = est.predict_survival_function(Xttest, return_array=False)

preds = [fn(36) for fn in surv]
times, rsfscore3y = brier_score(y, ytest, preds, 36)

preds = [fn(60) for fn in surv]
times, rsfscore5y = brier_score(y, ytest, preds, 60)

preds = [fn(120) for fn in surv]
times, rsfscore10y = brier_score(y, ytest, preds, 120)
outroc(ytest,rsf.predict(Xttest),"rsf")

rsfcindex=rsf.score(Xttest, ytest)

# estimate performance on training data, thus use `va_y` twice.
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)

va_auc, va_mean_auc = cumulative_dynamic_auc(y, ytest, cph.predict(Xttest0), va_times)
va_auc2, va_mean_auc2 = cumulative_dynamic_auc(y, ytest, estimator.predict(Xttest[non_zero_coefs.index.tolist()]), va_times)
va_auc3, va_mean_auc3 = cumulative_dynamic_auc(y, ytest, rsf.predict(Xttest), va_times)

plt.figure()
plt.plot(va_times, va_auc0, marker="o",markersize=3.,label ='cox')
plt.axhline(va_mean_auc, linestyle="--")

plt.plot(va_times, va_auc, marker="o",markersize=3.,label ='cox-en')
plt.axhline(va_mean_auc, linestyle="--")

plt.plot(va_times, va_auc2, marker="o",markersize=3.,label ='svm')
plt.axhline(va_mean_auc2, linestyle="--")

plt.plot(va_times, va_auc3, marker="o",markersize=3.,label='rsf')
plt.axhline(va_mean_auc3, linestyle="--")

plt.xlabel("months from enrollment")
plt.ylabel("time-dependent AUC")

plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 

plt.xlim(0,120)
plt.ylim(0.75,1)

plt.legend()
plt.grid(True)
#get the color
plt.rcParams['axes.prop_cycle'].by_key()['color']

print("rsfcindex=",rsfcindex)
print("rsfscore3y=",rsfscore3y)
print("rsfscore5y=",rsfscore5y)
print("rsfscore10y=",rsfscore10y)


#%% feature importance evaluated by RSF

import eli5
from eli5.sklearn import PermutationImportance
    
perm = PermutationImportance(rsf, n_iter=5, random_state=random_state)
perm.fit(Xt, y)
print("1")
feature_names = Xt.columns.tolist()
html_obj = eli5.show_weights(perm,feature_names=feature_names)

with open('C:\code\iris-importance2.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
import webbrowser
url = r'C:\code\iris-importance2.htm'
webbrowser.open(url, new=2)

#rsfimportance.xlsx was obtained from iris-importance2.htm
rsfimp=pd.read_excel(r'C:\code\rsfimportance.xlsx')
rsfimp.set_index(["Feature"], inplace=True)

_, ax = plt.subplots(figsize=(6, 8))
rsfimp.plot.barh(ax=ax, legend=False)
ax.set_xlabel("weight")
ax.set_ylabel("")
ax.grid(True)

#%%save the environment including the trained models
'''
import dill
filename= 'C:\code\globalsave.pkl'
dill.dump_session(filename)
#load the environment
import dill
filename= 'C:\code\globalsave.pkl'
dill.load_session(filename)
'''

































