from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
TAB = BASE / 'tables' / 'stage4r'
FIG = BASE / 'figures' / 'stage4r'
FIG.mkdir(parents=True, exist_ok=True)
bench = pd.read_csv(TAB/'PINUS_MODEL_BENCHMARK_SUMMARY.csv')
label = pd.read_csv(TAB/'PINUS_LABEL_EFFICIENCY_SUMMARY.csv')
guard = pd.read_csv(TAB/'PINUS_CONSENSUS_PERFORMANCE.csv')
risk = pd.read_csv(TAB/'PINUS_RISK_SUMMARY.csv')
conf = pd.read_csv(TAB/'PINUS_CONFORMAL_SUMMARY.csv')


def save_all(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG/f'{stem}.png', dpi=600, bbox_inches='tight')
    fig.savefig(FIG/f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(FIG/f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)

for target in ['MOE','MOR']:
    x = bench[(bench.target==target) & bench.model.isin(['frozen_zero_shot','density_ridge','full_hgb','full_extra_trees'])].copy()
    order=['frozen_zero_shot','density_ridge','full_hgb','full_extra_trees']
    x['model']=pd.Categorical(x.model,order,ordered=True);x=x.sort_values('model')
    labels=['Frozen zero-shot','Density Ridge','Full HGB','Full ExtraTrees']
    fig, ax = plt.subplots(figsize=(7.2,4.4))
    bars=ax.bar(labels,x.median_r2)
    ax.axhline(0, linewidth=.8)
    ax.set_ylabel('Median group-CV $R^2$')
    ax.set_title(f'Pinus {target}: transport failure and target-regime recovery')
    ax.tick_params(axis='x', rotation=18)
    for b,v in zip(bars,x.median_r2): ax.text(b.get_x()+b.get_width()/2,v+(0.025 if v>=0 else -0.07),f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
    save_all(fig,f'FIG13{target}_PINUS_TRANSPORT_OR_ADAPT_R2')

for target in ['MOE','MOR']:
    x=label[(label.target==target)&(label.model.isin(['density_ridge','full_hgb','full_extra_trees']))].copy()
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    for model,display in [('density_ridge','Density Ridge'),('full_hgb','Full HGB'),('full_extra_trees','Full ExtraTrees')]:
        g=x[x.model==model].sort_values('median_train_trees')
        ax.plot(g.median_train_trees,g.median_r2,marker='o',label=display)
        ax.fill_between(g.median_train_trees,g.q25_r2,g.q75_r2,alpha=.12)
    ax.axhline(0,linewidth=.8)
    ax.set_xlabel('Median number of labeled training trees')
    ax.set_ylabel('Median held-out-tree $R^2$')
    ax.set_title(f'Pinus {target}: tree-separated label efficiency')
    ax.legend(frameon=False)
    save_all(fig,f'FIG14{target}_PINUS_LABEL_EFFICIENCY')

fig,ax=plt.subplots(figsize=(7.2,4.4))
labels=['PREDICT','REVIEW','ABSTAIN'];x=np.arange(len(labels));width=.36
for offset,target in [(-width/2,'MOE'),(width/2,'MOR')]:
    g=guard[guard.target==target].set_index('decision').reindex(labels)
    ax.bar(x+offset,g.error_ratio,width,label=target)
ax.axhline(1,linewidth=.8)
ax.set_xticks(x,labels);ax.set_ylabel('Mean absolute-error ratio to target average')
ax.set_title('Pinus consensus guard separates low- and high-error records')
ax.legend(frameon=False)
save_all(fig,'FIG15_PINUS_CONSENSUS_GUARD_ERROR_RATIO')

fig,ax=plt.subplots(figsize=(7.2,4.4))
x=np.arange(2);width=.36
g90=conf[(conf.nominal==.9)&(conf.method.isin(['constant_crossfit','adaptive_learned_risk']))]
for offset,method,lab in [(-width/2,'constant_crossfit','Constant'),(width/2,'adaptive_learned_risk','Learned-risk adaptive')]:
    g=g90[g90.method==method].set_index('target').reindex(['MOE','MOR'])
    # plot normalized Winkler to compare targets with different units
    vals=[]
    for target in ['MOE','MOR']:
        base=g90[(g90.target==target)&(g90.method=='constant_crossfit')].mean_winkler.iloc[0]
        vals.append(g.loc[target,'mean_winkler']/base)
    ax.bar(x+offset,vals,width,label=lab)
ax.axhline(1,linewidth=.8);ax.set_xticks(x,['MOE','MOR']);ax.set_ylabel('Winkler loss / constant baseline')
ax.set_title('Pinus 90% conformal efficiency: adaptive versus constant')
ax.legend(frameon=False)
save_all(fig,'FIG16_PINUS_CONFORMAL_RELATIVE_WINKLER')
print(FIG)
