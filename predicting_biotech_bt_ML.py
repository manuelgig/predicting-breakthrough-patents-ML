# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:54:06 2021

@author: Manuel G.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA

# level for the radical indicators: application/docdb/inpadoc
level_list = ['docdb','inpadoc']
level = 0

# lists with control variables
controls = ['applt_cnt', 'invt_cnt','nr_bw_cites', 'nr_npl_cites', 'nr_claims', 'nr_ipc4', 'nr_ipc6'] 
time_dummies = list(range(1981,2002))
existing_exante = ['original_ipc4', 'original_ipc6', 'shane_ipc3', 'shane_ipc4', 'shane_ipc6']

# lists with radical indicators
nto_dummy = ['nto4', 'nto6']
nso_dummy = ['nso4', 'nso6']
nf_dummy = ['nf4', 'nf6']
nto_count = ['nto4_count', 'nto6_count']
nso_count= ['nso4_count', 'nso6_count']
nf_count = ['nf4_count', 'nf6_count']
nto_corr = ['nto4_corr', 'nto6_corr']
nso_corr = ['nso4_corr', 'nso6_corr']
nf_corr = ['nf4_corr', 'nf6_corr']
nto_all = nto_dummy + nto_count + nto_corr
nso_all = nso_dummy + nso_count + nso_corr
nf_all = nf_dummy + nf_count + nf_corr
new_rad = ['top_avg_md_sd3']

rad_dumm = nto_dummy + nso_dummy + nf_dummy #+ new_rad
rad_count = nto_count + nso_count + nf_count #+ new_rad
rad_corr = nto_corr + nso_corr + nf_corr #+new_rad
rad_all = nto_all + nso_all + nf_all #+ new_rad

scalable = controls + existing_exante + nto_count + nso_count + nf_count + new_rad
    
def load_df(level):
    """ Laods and returns the desired dataset (docdb/inpadoc) """
    df = pd.read_csv('01_biotech_val_data_{}.csv'.format(level))
    
    return df
    
def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    """Compute Receiver operating characteristic (ROC)

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        Label considered as positive and others are considered negative.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

        .. versionadded:: 0.17
           parameter *drop_intermediate*.

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([ 0. ,  0.5,  0.5,  1. ])
    >>> tpr
    array([ 0.5,  0.5,  1. ,  1. ])
    >>> thresholds
    array([ 0.8 ,  0.4 ,  0.35,  0.1 ])

    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds

def benchmark_plot():
    df = pd.DataFrame.from_records([['Baseline']*4+['RFC']*4,
                                    ['Recall','Precision','Average precision','AUC']*2,
                                    [0.174,0.872,0.297,0.587,0.693,1.000,0.872,0.847]], index=['Model','Indicator','Value']).T
    plt.figure()
    sns.barplot(data=df,x='Indicator',y='Value',hue='Model')
    ax = plt.gca()
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    ax.set_title('Figure 1. Performance benchmark: Baseline vs RFC models')
    plt.tight_layout()
    
def eda_1():
    for i in ['nto4','nso4','nf4','nto6','nso6','nf6']:
        df.loc[df['{}_count'.format(i)]==0,'{}_count_cat'.format(i)] = '0'
        df.loc[df['{}_count'.format(i)]>0,'{}_count_cat'.format(i)] = '1-3'
        df.loc[df['{}_count'.format(i)]>3,'{}_count_cat'.format(i)] = '>=4'
    
    rad_dummies = ['nto4_count_cat', 'nso4_count_cat', 'nf4_count_cat']
    titles = ['A. nto4', 'B. nso4', 'C. nf4']
    fig, axes = plt.subplots(1,3, sharey='row',figsize=(10,5))
    for i in range(3):
        sns.barplot(data=df, x=rad_dummies[i], y='important', ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_ylabel('Breakthrough rate')
        if i==1:
            axes[i].set_xlabel('Score')
        else:
            axes[i].set_xlabel('')
        axes[i].yaxis.set_tick_params(labelbottom=True)
    fig.suptitle('Figure 2. Breakthrough rate at different values of NTO4/NSO4/NF4', y=0.98)
    fig.subplots_adjust(top=0.8)
    plt.tight_layout()
    
def bt_rates():
    ### Table with breakthrough rates
    print('Table 1. Breakthrough rate for subsets with different scores of NTO4, NSO4 and NF4')
    pd.options.display.float_format = '{:,.3f}'.format
    print(pd.DataFrame(df[['nto4_count_cat','important']].rename(columns={'nto4_count_cat':'score','important':'nto4'}).groupby('score')['nto4'].mean()).merge(
    df[['nso4_count_cat','important']].rename(columns={'nso4_count_cat':'score','important':'nso4'}).groupby('score')['nso4'].mean(),
    left_on='score',right_index=True).merge(
    df[['nf4_count_cat','important']].rename(columns={'nf4_count_cat':'score','important':'nf4'}).groupby('score')['nf4'].mean(),
    left_on='score',right_index=True))
        

def eda_2():
    df['comp_dum4'] = df['nto4'] + df['nso4'] + df['nf4']
    df['comp_dum4'] = df['nto4'] + df['nso4'] + df['nf4']
    df.loc[df['nto4_count'] + df['nso4_count'] + df['nf4_count']==0,'comp_dum4_cat'] = '0'
    df.loc[df['nto4_count'] + df['nso4_count'] + df['nf4_count']>0,'comp_dum4_cat'] = '1-3'
    df.loc[df['nto4_count'] + df['nso4_count'] + df['nf4_count']>3,'comp_dum4_cat'] = '>=4'
    
    fig, axes = plt.subplots(1,2, sharey='row', figsize=(10, 5))
    sns.barplot(data=df, x='comp_dum4', y='important', ax=axes[0])
    sns.barplot(data=df, x='comp_dum4_cat', y='important', ax=axes[1])
    axes[0].set_title('A. Composite4')
    axes[1].set_title('B. Composite score')
    axes[1].yaxis.set_tick_params(labelbottom=True)
    for i in [0,1]:
        axes[i].set_ylabel('Breakthrough rate')
        axes[i].set_xlabel('')
    fig.suptitle('Figure 3. Scoring in multiple features', y=0.98)
    fig.subplots_adjust(top=0.8)
    plt.tight_layout()
    
def model(rad_levels, features_order, feature_names, model_features, cutoff=.5, scaler='MinMaxScaler', plot=True, label='important', penalty='l2', condensed=True, solver='liblinear',lambda_=1, l1_ratio=None):
    global plots, outc
    
    if all(item in ['appln','docdb','inpadoc'] for item in rad_levels)==False:
        print("""Unrecognized value for rad_levels. Valid values are 'appln','docdb' and 'inpadoc'""")
        return    
   
    if 'StandardScaler' not in ['StandardScaler','MinMaxScaler']:
        print("""Unrecognized value for scaler. Valid values are None, 'MinMaxScaler' and 'StandardScaler'""")
        return

    outcomes = []
    
    scalers = {'MinMaxScaler':MinMaxScaler(),'StandardScaler':StandardScaler()}
    
    m=0
    
    if plot==True:
        models  = []
        
        fig, axs = plt.subplots(1, 2,figsize=(12,5))
        st = fig.suptitle('Figure 4. Model comparison (regularization={}, lambda={})'.format(penalty,lambda_), fontsize="x-large")
        #st.set_y(0.90)
        
        axs[0].plot([0, 1], [0, 1], 'k--')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('ROC Curve')
        
        axs[1].set_title('2-class Precision-Recall curve')
        
    for level in rad_levels:

        df = load_df(level)
        y = df[label]
        df['patage'] = df['appy'] - df['appy'].min()
        list_ipc4 = [x for x in df.columns if x[:6]=='ipc4__']
        df = df.merge(pd.get_dummies(df['appy']), left_index=True,right_index=True)
        
        transformer = FunctionTransformer(np.log1p, validate=True)
        df[scalable] = transformer.transform(df[scalable])
        
        if scaler in ['MinMaxScaler','StandardScaler']:
            #print('Escalando con {}'.format(scaler))
            scaled = scalers[scaler].fit(df[scalable])
            df[scalable] = scaled.transform(df[scalable])
        
        for vect in features_order:
            
            if (condensed==True) & (vect==[1,1,1,0]) & (level=='inpadoc'):
                continue
            
            logreg = LogisticRegression(max_iter=1000, penalty=penalty, solver=solver, C=lambda_, l1_ratio=l1_ratio)        
            
            lis = []
            
            model_desc = vect
            model_desc = model_desc + ['No' if level=='docdb' else 'Yes'.format(level) for level in [level]]

            for i in range(len(vect)):
                if vect[i]:
                    lis = lis + model_features[i]
                    
            X = df[lis]
            logreg.fit(X, y)
            
            y_pred = np.array([1 if x>=cutoff else 0 for x in logreg.predict_proba(X)[:,1]])
            
            y_score = logreg.decision_function(X)
            average_precision = average_precision_score(y, y_score)            
            false_positive_rate_test, true_positive_rate_test, thresholds_test = roc_curve(y, y_pred)
            roc_auc_test = auc(false_positive_rate_test, true_positive_rate_test)
            
            model_desc = model_desc + [penalty,lambda_, recall_score(y,y_pred), precision_score(y, y_pred),   accuracy_score(y, y_pred),  average_precision, roc_auc_test]
        
            outcomes.append(model_desc)            
            
            if plot==True:                                
                y_pred_prob = logreg.predict_proba(X)[:,1]
                fpr, tpr, thresholds = roc_curve(y, y_pred_prob)        
                axs[0].plot(fpr, tpr, label='Model {}'.format(m+1))
                axs[0].legend()
                #plots.append([fpr,tpr,thresholds])                

                plot_precision_recall_curve(logreg,X,y, ax=axs[1])        
                
            m+=1
            
    outcomes_df = pd.DataFrame(outcomes).T
    outcomes_df.columns = ['Model {}'.format(x+1) for x in range(outcomes_df.shape[1])]
    outcomes_df.iloc[0:len(feature_names),:] = outcomes_df.iloc[0:len(feature_names),:].replace({0:'No',1:'Yes'})
    outcomes_df.index =  feature_names + ['Inpadoc fam. correction'] + ['Penalty','Lambda','Recall (cutoff={})'.format(cutoff),'Precision (cutoff={})'.format(cutoff),'Accuracy','Average precision', 'AUC']    
        
    return outcomes_df

def train_test_validation(level='inpadoc', penalty='l2', solver='liblinear', lambda_=1, random_state=150, scaler='MinMaxScaler'):
    
    if scaler not in [None,'MinMaxScaler','StandardScaler']:
        return print('Valor de scaler incorrecto. Scalers posibles son None, MinMaxScaler o StandarScaler')
    
    df = load_df(level)
    df['patage'] = df['appy'] - df['appy'].min()
    list_ipc4 = [x for x in df.columns if x[:6]=='ipc4__']
    df = df.merge(pd.get_dummies(df['appy']), left_index=True,right_index=True)    
    transformer = FunctionTransformer(np.log1p, validate=True)
    df[scalable] = transformer.transform(df[scalable])    
    X = df[list_ipc4 + controls + time_dummies + existing_exante + rad_all]
    y = df['important']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=random_state)      
    
    if scaler in ['MinMaxScaler','StandardScaler']:
        scalers = {'MinMaxScaler':MinMaxScaler(),'StandardScaler':StandardScaler()}
        #print('Escalando con {}'.format(scaler))
        X_train = scalers[scaler].fit_transform(X_train)
        X_test = scalers[scaler].transform(X_test)
    
    logreg = LogisticRegression(max_iter=1000, penalty=penalty, solver='liblinear', C=lambda_)        
    logreg.fit(X_train, y_train)
    
    y_test_pred = logreg.predict(X_test)
    y_train_pred = logreg.predict(X_train)
    y_test_score = logreg.decision_function(X_test)
    average_precision_test = average_precision_score(y_test, y_test_score)            
    y_train_score = logreg.decision_function(X_train)
    average_precision_train = average_precision_score(y_train, y_train_score)            
    
    false_positive_rate_test, true_positive_rate_test, thresholds_test = roc_curve(y_test, y_test_pred)
    roc_auc_test = auc(false_positive_rate_test, true_positive_rate_test)
    false_positive_rate_train, true_positive_rate_train, thresholds_train = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(false_positive_rate_train, true_positive_rate_train)    
    
    res = [['Yes','Yes']]*5
    res.append([round(recall_score(y_train,y_train_pred),3),round(recall_score(y_test,y_test_pred),3)])
    res.append([round(precision_score(y_train, y_train_pred),3), round(precision_score(y_test, y_test_pred),3)])
    res.append([round(accuracy_score(y_train, y_train_pred),3), round(accuracy_score(y_test, y_test_pred),3)])
    res.append([round(average_precision_train,3), round(average_precision_test,3)])
    res.append([round(roc_auc_train,3),round(roc_auc_test,3)])
    res = pd.DataFrame.from_records(res,columns=['Train','Test'],
                                    index=['IPC4', 'Controls','Existing ex-ante','Novelty','INPADOC fam. correction',
                                           'Recall','Precision','Accuracy','Avg precision','AUC'])
    
    #print('Confusion matrix','\n',confusion_matrix(y_test, y_test_pred))
    #print(classification_report(y_test, y_test_pred))
    
    m=0
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)        
    fig, axs = plt.subplots(1, 2,figsize=(12,5))
    st = fig.suptitle('Figure 5. Model 3 - Train/test validation split (regularization={}, lambda={})'.format(penalty,lambda_), fontsize="x-large")
    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curve')
    axs[0].plot(fpr, tpr, label='Test set')
    plot_precision_recall_curve(logreg,X_test,y_test, ax=axs[1])        
    axs[1].set_title('2-class Precision-Recall curve')
    
    m=1
    y_pred_prob = logreg.predict_proba(X_train)[:,1]
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob)        
    axs[0].plot(fpr, tpr, label='Train set')
    plot_precision_recall_curve(logreg,X_train,y_train, ax=axs[1])
    axs[0].legend()      
    
    print('Table 3. Train/test split validation for Model 3\n')
    print(res)
    #return res

def rfc_1(level):
    df = load_df(level)
    df['patage'] = df['appy'] - df['appy'].min()
    list_ipc4 = [x for x in df.columns if x[:6]=='ipc4__']
    df = df.merge(pd.get_dummies(df['appy']), left_index=True,right_index=True)
    
    X = df[list_ipc4 + controls + time_dummies + existing_exante + rad_all]
    y = df['important']
    
    random_state=20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=random_state)      
    
    scaler='MinMaxScaler'
    scalers = {'MinMaxScaler':MinMaxScaler(),'StandardScaler':StandardScaler()}
    
    rf = RandomForestClassifier(n_estimators=20, random_state=random_state, max_depth=25, max_features=30, min_samples_leaf=1, criterion='entropy')
    rf.fit(X_train,y_train)
    
    rf.score(X_train,y_train)
    rf.score(X_test,y_test)
    
    y_test_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)
    rmse_test = MSE(y_test,y_test_pred)**(1/2)
    rmse_train = MSE(y_train,y_train_pred)**(1/2)
    #print(rmse_train,rmse_test)
    
    y_score = rf.predict_proba(X_test)[:,1]
    average_precision_test = average_precision_score(y_test, y_score)
    y_score_train = rf.predict_proba(X_train)[:,1]
    average_precision_train = average_precision_score(y_train, y_score_train)
    
    false_positive_rate_test, true_positive_rate_test, thresholds_test = roc_curve(y_test, y_test_pred)
    roc_auc_test = auc(false_positive_rate_test, true_positive_rate_test)
    false_positive_rate_train, true_positive_rate_train, thresholds_train = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(false_positive_rate_train, true_positive_rate_train)
    
    res = [['Yes','Yes']]*5
    res.append([round(recall_score(y_train,y_train_pred),3),round(recall_score(y_test,y_test_pred),3)])
    res.append([round(precision_score(y_train, y_train_pred),3), round(precision_score(y_test, y_test_pred),3)])
    res.append([round(accuracy_score(y_train, y_train_pred),3), round(accuracy_score(y_test, y_test_pred),3)])
    res.append([round(average_precision_train,3), round(average_precision_test,3)])
    res.append([round(roc_auc_train,3),round(roc_auc_test,3)])
    res = pd.DataFrame.from_records(res,columns=['Train','Test'],
                                    index=['IPC4', 'Controls','Existing ex-ante','Novelty','INPADOC fam. correction',
                                           'Recall','Precision','Accuracy','Avg precision','AUC'])
    
    #print('Table 4. RFC - confusion matrix\n','\n',confusion_matrix(y_test, y_test_pred))
    print('Table 4. RFC - Quality indicators\n','\n', res)
    
    disp = plot_precision_recall_curve(rf, X_test, y_test)
    disp.ax_.set_title('Figure 6. RFC 2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision_test));

    X_colnames = X.columns
    return rf, X_colnames

def rfc_2(fitted_rfc, X_colnames):
    # most important features
    feature_imp = pd.Series(fitted_rfc.feature_importances_,index=X_colnames).sort_values(ascending=False)
    feature_imp2 = feature_imp[:30]
    plt.figure(figsize=(8,8))
    sns.barplot(x=feature_imp2, y=feature_imp2.index, color='steelblue')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Figure 7. Visualizing the most important features')
    #plt.legend()
    plt.show()
    plt.tight_layout()
