#code
from model.your_model import predict
from conf.conf import logging, settings


pred = predict(settings.PREDICT.test, model=settings.MODEL.top)
logging.info(f"Prediction - {pred}")