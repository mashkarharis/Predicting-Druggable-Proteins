import sys
import logging

# Create Logger
logger = logging.getLogger(__name__)

# First read the arguments
logger.info("Reading Arguments :")
TR_POS = sys.argv[1]
TR_NEG = sys.argv[2]
TS_POS = sys.argv[3]
TS_NEG = sys.argv[4]

logger.info('- TR_POS'+str(TR_POS))
logger.info('- TR_NEG'+str(TR_NEG))
logger.info('- TS_POS'+str(TS_POS))
logger.info('- TS_NEG'+str(TS_NEG))
logger.info('')



