from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

class ModelTrainer:

    def __init__(self,logger) -> None:
        self.logger  = logger

    def train_RSSECOND(self,tr_X,tr_y,ts_X,ts_y):
        
        model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, solver='lbfgs', max_iter=100)

        model.fit(tr_X,tr_y)
        y_pred = model.predict(ts_X)
        self.printResult(ts_y,y_pred)
        return model

    def train_RSDHP(self,tr_X,tr_y,ts_X,ts_y):
        
        model = svm.SVC(C=1.3,kernel='linear',gamma='auto',shrinking= True,decision_function_shape="ovr")
        model.fit(tr_X,tr_y)
        y_pred = model.predict(ts_X)
        self.printResult(ts_y,y_pred)
        return model
    
    def train_RSACID(self,tr_X,tr_y,ts_X,ts_y):
        
        model = LinearSVC(penalty='l2', loss='squared_hinge', C=20.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, max_iter=3000)
        model.fit(tr_X,tr_y)
        y_pred = model.predict(ts_X)
        self.printResult(ts_y,y_pred)
        return model
    
    def train_RSPOLAR(self,tr_X,tr_y,ts_X,ts_y):
        
        model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, solver='lbfgs', max_iter=100)
        model.fit(tr_X,tr_y)
        y_pred = model.predict(ts_X)
        self.printResult(ts_y,y_pred)
        return model
    
    def train_RSCHARGE(self,tr_X,tr_y,ts_X,ts_y):
        
        model = ExtraTreesClassifier(n_estimators=100,criterion='gini',min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0)
        model.fit(tr_X,tr_y)
        y_pred = model.predict(ts_X)
        self.printResult(ts_y,y_pred)
        return model

    def printResult(self,y_test,y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        self.logger.info('  Accuracy:'+str(accuracy))
        self.logger.info('  Sensitivity:'+str(sensitivity))
        self.logger.info('  Specificity:'+str(specificity))
        self.logger.info('  Precision:'+str(precision))
        self.logger.info('  F1-score:'+str(f1))
        self.logger.info('  MCC:'+str(mcc))