# here is pickle 
import pickle
from conf.conf import logging, settings

def save_model(model, rf:bool = True) -> None:
    if rf:
        logging.info('saving random forest')
        pickle.dump(model, open(settings.MODEL.rf_path, 'wb'))
        logging.info('saving random forest - success')
    else:
        logging.info('saving gradient boosting')
        pickle.dump(model, open(settings.MODEL.gb_path, 'wb'))
        logging.info('saving gradient boosting - success')

def load_model(path:str) -> None:
    logging.info('loading model')
    clf = pickle.load(open(settings.MODEL.rf_path, 'rb'))
    logging.info('loading model - success')
    return clf
    
