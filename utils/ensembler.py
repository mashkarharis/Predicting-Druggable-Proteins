import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

class Ensembler:

    def __init__(self,logger) -> None:
        self.logger=logger

    def Majority_Vote_Ensembler(self,models,val_data,test_data):
        
        self.logger.info(" Calculating Predictions")
        y_pred_model1 = models["RSSECOND"].predict(test_data["RSSECOND"][0])
        y_pred_model2 = models["RSDHP"].predict(test_data["RSDHP"][0])
        y_pred_model3 = models["RSACID"].predict(test_data["RSACID"][0])
        y_pred_model4 = models["RSPOLAR"].predict(test_data["RSPOLAR"][0])
        y_pred_model5 = models["RSCHARGE"].predict(test_data["RSCHARGE"][0])

        self.logger.info(" Choosing Majority Votes")
        y_pred_ensemble = mode([y_pred_model1, y_pred_model2, y_pred_model3, y_pred_model4, y_pred_model5], axis=0,keepdims=False).mode.flatten()
        
        self.logger.info("")
        self.printResult(test_data["RSSECOND"][1],y_pred_ensemble)
    
    def Weighted_Vote_Ensembler(self,models,val_data,test_data):
        
        # Predict For Train Data
        self.logger.info(" Get Train Data Prediction For Each Model")
        y_pred_model1_train = models["RSSECOND"].predict(val_data["RSSECOND"][0])
        y_pred_model2_train = models["RSDHP"].predict(val_data["RSDHP"][0])
        y_pred_model3_train = models["RSACID"].predict(val_data["RSACID"][0])
        y_pred_model4_train = models["RSPOLAR"].predict(val_data["RSPOLAR"][0])
        y_pred_model5_train = models["RSCHARGE"].predict(val_data["RSCHARGE"][0])

        # Stack Train Predictons
        meta_features_train = np.column_stack((y_pred_model1_train, y_pred_model2_train, y_pred_model3_train,y_pred_model4_train,y_pred_model5_train))

        # Train A Ensembler
        meta_classifier = LogisticRegression(random_state=42)
        meta_classifier.fit(meta_features_train, val_data["RSCHARGE"][1])

        # Contributions
        self.logger.info("")
        contributions = meta_classifier.coef_
        contributions_percentages = (contributions / np.sum(contributions)) * 100
        for i, classifier in enumerate(['RSSECOND', 'RSDHP', 'RSACID',"RSPOLAR","RSCHARGE"]):
            self.logger.info(f" Contribution of {classifier}: {contributions_percentages[0][i]:.2f}%")
        
        # Now Testing
        self.logger.info("")
        y_pred_model1 = models["RSSECOND"].predict(test_data["RSSECOND"][0])
        y_pred_model2 = models["RSDHP"].predict(test_data["RSDHP"][0])
        y_pred_model3 = models["RSACID"].predict(test_data["RSACID"][0])
        y_pred_model4 = models["RSPOLAR"].predict(test_data["RSPOLAR"][0])
        y_pred_model5 = models["RSCHARGE"].predict(test_data["RSCHARGE"][0])
        meta_features_test = np.column_stack((y_pred_model1, y_pred_model2, y_pred_model3,y_pred_model4,y_pred_model5))
        y_pred_ensemble = meta_classifier.predict(meta_features_test)
        self.logger.info("")
        self.printResult(test_data["RSSECOND"][1],y_pred_ensemble)



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