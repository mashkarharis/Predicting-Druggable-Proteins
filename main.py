import sys
import logging
import warnings
import pandas as pd
from utils.file_manager import FileManager
from utils.feature_extractor import FeatureExtractor
from utils.model_train import ModelTrainer
from utils.ensembler import Ensembler
from sklearn.model_selection import train_test_split

# Create Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# First read the arguments
logger.info("Reading Arguments :")
TR_POS = sys.argv[1]
TR_NEG = sys.argv[2]
TS_POS = sys.argv[3]
TS_NEG = sys.argv[4]

logger.info(' TR_POS'+str(TR_POS))
logger.info(' TR_NEG'+str(TR_NEG))
logger.info(' TS_POS'+str(TS_POS))
logger.info(' TS_NEG'+str(TS_NEG))
logger.info('')

# Read Files and Convert FASTA TO CSV
logger.info('Convert FAST to DataFrames')
file_manager = FileManager()

TR_POS_DF = file_manager.convert_fasta_to_df(TR_POS)
TR_POS_DF['label']=1
TR_NEG_DF = file_manager.convert_fasta_to_df(TR_NEG)
TR_NEG_DF['label']=0
TS_POS_DF = file_manager.convert_fasta_to_df(TS_POS)
TS_POS_DF['label']=1
TS_NEG_DF = file_manager.convert_fasta_to_df(TS_NEG)
TS_NEG_DF['label']=0

logger.info('TR_POS_DF SHAPE : '+str(TR_POS_DF.shape))
logger.info('TR_NEG_DF SHAPE : '+str(TR_NEG_DF.shape))
logger.info('TS_POS_DF SHAPE : '+str(TS_POS_DF.shape))
logger.info('TS_NEG_DF SHAPE : '+str(TS_NEG_DF.shape))
logger.info('')

# Concat DataFrames
logger.info('Merge Frames')
TR_ALL_DF = pd.concat([TR_POS_DF, TR_NEG_DF])
TR_ALL_DF = TR_ALL_DF.sample(frac=1).reset_index(drop=True)
logger.info(' Training Shape : '+str(TR_ALL_DF.shape))
TS_ALL_DF = pd.concat([TS_POS_DF, TS_NEG_DF])
TS_ALL_DF = TS_ALL_DF.sample(frac=1).reset_index(drop=True)
logger.info(' Testing Shape : '+str(TS_ALL_DF.shape))
logger.info('')

# Extract Features
logger.info('Extract Features')
feature_extractor = FeatureExtractor()

#RSSECOND
rssecond_train_X,rssecond_train_y = feature_extractor.extract_RSSECOND(TR_ALL_DF)
rssecond_test_X,rssecond_test_y = feature_extractor.extract_RSSECOND(TS_ALL_DF)
logger.info(' RSSECOND :'+str(rssecond_train_X.shape)+", "+str(rssecond_train_y.shape)+", "+str(rssecond_test_X.shape)+", "+str(rssecond_test_y.shape))

# RHDP
rdhp_train_X,rdhp_train_y = feature_extractor.extract_RDHP(TR_ALL_DF)
rdhp_test_X,rdhp_test_y = feature_extractor.extract_RDHP(TS_ALL_DF)
logger.info(' RSDHP :'+str(rdhp_train_X.shape)+", "+str(rdhp_train_y.shape)+", "+str(rdhp_test_X.shape)+", "+str(rdhp_test_y.shape))

#RSACID
rsacid_train_X,rsacid_train_y = feature_extractor.extract_RSACID(TR_ALL_DF)
rsacid_test_X,rsacid_test_y = feature_extractor.extract_RSACID(TS_ALL_DF)
logger.info(' RSACID :'+str(rsacid_train_X.shape)+", "+str(rsacid_train_y.shape)+", "+str(rsacid_test_X.shape)+", "+str(rsacid_test_y.shape))

#RSPOLAR
rspolar_train_X,rspolar_train_y = feature_extractor.extract_RSPOLAR(TR_ALL_DF)
rspolar_test_X,rspolar_test_y = feature_extractor.extract_RSPOLAR(TS_ALL_DF)
logger.info(' RSPOLAR :'+str(rspolar_train_X.shape)+", "+str(rspolar_train_y.shape)+", "+str(rspolar_test_X.shape)+", "+str(rspolar_test_y.shape))

#RSCHARGE
rscharge_train_X,rscharge_train_y = feature_extractor.extract_RSCHARGE(TR_ALL_DF)
rscharge_test_X,rscharge_test_y = feature_extractor.extract_RSCHARGE(TS_ALL_DF)
logger.info(' RSCHARGE :'+str(rscharge_train_X.shape)+", "+str(rscharge_train_y.shape)+", "+str(rscharge_test_X.shape)+", "+str(rscharge_test_y.shape))

# Training Model
logger.info('')
logger.info('Training Models')
model_trainer = ModelTrainer(logger)

logger.info(' RSSECOND')
rssecond_model = model_trainer.train_RSSECOND(rssecond_train_X,rssecond_train_y,rssecond_test_X,rssecond_test_y)
logger.info('  Best Model : '+str(rssecond_model))
logger.info('')

logger.info(' RSDHP')
rsdhp_model = model_trainer.train_RSDHP(rdhp_train_X,rdhp_train_y,rdhp_test_X,rdhp_test_y)
logger.info('  Best Model : '+str(rsdhp_model))
logger.info('')

logger.info(' RSACID')
rsacid_model = model_trainer.train_RSACID(rsacid_train_X,rsacid_train_y,rsacid_test_X,rsacid_test_y)
logger.info('  Best Model : '+str(rsacid_model))
logger.info('')

logger.info(' RSPOLAR')
rspolar_model = model_trainer.train_RSPOLAR(rspolar_train_X,rspolar_train_y,rspolar_test_X,rspolar_test_y)
logger.info('  Best Model : '+str(rspolar_model))
logger.info('')

logger.info(' RSCHARGE')
rscharge_model = model_trainer.train_RSCHARGE(rscharge_train_X,rscharge_train_y,rscharge_test_X,rscharge_test_y)
logger.info('  Best Model : '+str(rscharge_model))
logger.info('')

# Define Data
models = {
    "RSSECOND":rssecond_model,
    "RSDHP":rsdhp_model,
    "RSACID":rsacid_model,
    "RSPOLAR":rspolar_model,
    "RSCHARGE":rscharge_model
}

test_data = {
    "RSSECOND":[rssecond_test_X,rssecond_test_y],
    "RSDHP":[rdhp_test_X,rdhp_test_y],
    "RSACID":[rsacid_test_X,rsacid_test_y],
    "RSPOLAR":[rspolar_test_X,rspolar_test_y],
    "RSCHARGE":[rscharge_test_X,rscharge_test_y]
}

# Divide To Test And Validation
logger.info("Val / Test Splitting . . . ")
test_size = 0.2  
val_data, new_test_data = {}, {}
for key, value in test_data.items():
    X, y = value
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    val_data[key] = [X_val, y_val]
    new_test_data[key] = [X_test, y_test]
logger.info(" Val Size :"+str(len(val_data["RSCHARGE"][0])))
logger.info(" Test Size :"+str(len(new_test_data["RSCHARGE"][0])))
logger.info("")

# Ensembling
logger.info('Ensembling [1] : Majority Hard Voted Cassifier')
ensembler = Ensembler(logger)
ensembler.Majority_Vote_Ensembler(models,val_data,new_test_data)
logger.info('')

logger.info('Ensembling [2] : Weighted Voted Cassifier')
ensembler = Ensembler(logger)
ensembler.Weighted_Vote_Ensembler(models,val_data,new_test_data)
logger.info('')
