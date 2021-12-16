# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 23:31:20 2021

@author: xiao
"""
#%% KM curves

import csv
coxkm = pd.read_csv(r'C:\code\coxforkm.csv')
cox2km = pd.read_csv(r'C:\code\cox2forkm.csv')
svmkm = pd.read_csv(r'C:\code\svmforkm.csv')
rsfkm = pd.read_csv(r'C:\code\rsfforkm.csv')

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

# KM curves for the high and low risk group of cox-en
E=coxkm[coxkm['subtype']=="low"]['os']
T=coxkm[coxkm['subtype']=="low"]['time']
Etest=coxkm[coxkm['subtype']=="high"]['os']
Ttest=coxkm[coxkm['subtype']=="high"]['time']

fig = plt.figure(figsize=(6, 5.5))
ax = plt.subplot(111)
ax.set_xlim(-1,)

kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(T, event_observed=E, label="Low risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

kmf_exp = KaplanMeierFitter()
ax = kmf_exp.fit(Ttest, event_observed=Etest, label="High risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

from lifelines.statistics import logrank_test
results = logrank_test(T, Ttest, E, Etest, alpha=.99)
p='%.3f' % results.p_value
from lifelines.plotting import add_at_risk_counts

plt.rcParams['savefig.dpi'] = 600 
plt.rcParams['figure.dpi'] = 300

ax.set_xlabel('Survival time(months)',fontsize=15)
plt.text(90, 0.05, "Logrank Test: P<.001", size = 15)
plt.ylim(0,1) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)

plt.tight_layout()
fig.savefig(r"C:\code\coxkm.jpg")

# KM curves for the high and low risk group of cox
E=cox2km[cox2km['subtype']=="low"]['os']
T=cox2km[cox2km['subtype']=="low"]['time']
Etest=cox2km[cox2km['subtype']=="high"]['os']
Ttest=cox2km[cox2km['subtype']=="high"]['time']

fig = plt.figure(figsize=(6, 5.5))

ax = plt.subplot(111)
ax.set_xlim(-1,)

kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(T, event_observed=E, label="Low risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))
kmf_exp = KaplanMeierFitter()
ax = kmf_exp.fit(Ttest, event_observed=Etest, label="High risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

from lifelines.statistics import logrank_test
results = logrank_test(T, Ttest, E, Etest, alpha=.99)

p='%.3f' % results.p_value

from lifelines.plotting import add_at_risk_counts

plt.rcParams['savefig.dpi'] = 600 
plt.rcParams['figure.dpi'] = 300 
ax.set_xlabel('Survival time(months)',fontsize=15)
plt.text(90, 0.05, "Logrank Test: P<.001", size = 15)
plt.ylim(0,1) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
fig.savefig(r"C:\code\cox2km.jpg")

# KM curves for the high and low risk group of svm
E=svmkm[svmkm['subtype']=="low"]['os']
T=svmkm[svmkm['subtype']=="low"]['time']
Etest=svmkm[svmkm['subtype']=="high"]['os']
Ttest=svmkm[svmkm['subtype']=="high"]['time']

fig = plt.figure(figsize=(6, 5.5))
ax = plt.subplot(111)
ax.set_xlim(-1,)

kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(T, event_observed=E, label="Low risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

kmf_exp = KaplanMeierFitter()
ax = kmf_exp.fit(Ttest, event_observed=Etest, label="High risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

from lifelines.statistics import logrank_test
results = logrank_test(T, Ttest, E, Etest, alpha=.99)

p='%.3f' % results.p_value

from lifelines.plotting import add_at_risk_counts

plt.rcParams['savefig.dpi'] = 600 
plt.rcParams['figure.dpi'] = 300 


ax.set_xlabel('Survival time(months)',fontsize=15)
plt.text(90, 0.05, "Logrank Test: P<.001", size = 15)
plt.ylim(0,1) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
fig.savefig(r"C:\code\svmkm.jpg")

# KM curves for the high and low risk group of rsf
E=rsfkm[rsfkm['subtype']=="low"]['os']
T=rsfkm[rsfkm['subtype']=="low"]['time']

Etest=rsfkm[rsfkm['subtype']=="high"]['os']
Ttest=rsfkm[rsfkm['subtype']=="high"]['time']

fig = plt.figure(figsize=(6, 5.5))
ax = plt.subplot(111)
ax.set_xlim(-1,)

kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(T, event_observed=E, label="Low risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))
kmf_exp = KaplanMeierFitter()
ax = kmf_exp.fit(Ttest, event_observed=Etest, label="High risk").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

from lifelines.statistics import logrank_test
results = logrank_test(T, Ttest, E, Etest, alpha=.99)

p='%.3f' % results.p_value

from lifelines.plotting import add_at_risk_counts
plt.rcParams['savefig.dpi'] = 600 
plt.rcParams['figure.dpi'] = 300 
ax.set_xlabel('Survival time(months)',fontsize=15)
plt.text(90, 0.05, "Logrank Test: P<.001", size = 15)
plt.ylim(0,1) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
fig.savefig(r"C:\code\rsfkm.jpg")

# KM curves for training and test set

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

E=y['OS']
T=y['OS_Mon']

Etest=ytest['OS']
Ttest=ytest['OS_Mon']

fig = plt.figure(figsize=(6, 5.5))
ax = plt.subplot(111)

ax.set_xlim(-1,)

kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(T, event_observed=E, label="Training").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

kmf_exp = KaplanMeierFitter()
ax = kmf_exp.fit(Ttest, event_observed=Etest, label="Test").plot_survival_function(ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

from lifelines.statistics import logrank_test
results = logrank_test(T, Ttest, E, Etest, alpha=.99)

p='%.2f' % results.p_value

from lifelines.plotting import add_at_risk_counts

add_at_risk_counts(kmf_control,kmf_exp, ax=ax,xticks=np.linspace(0,180,10,endpoint=True))

plt.rcParams['savefig.dpi'] = 600 
plt.rcParams['figure.dpi'] = 300 
ax.set_xlabel('Survival time(months)')
plt.text(110, 0.1, "Logrank Test: P="+p, size = 10)
plt.ylim(0,1) 
plt.tight_layout()

fig.savefig(r"C:\code\train and test.jpg")



