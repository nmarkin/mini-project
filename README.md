# mini-project
HW for research seminar


The project structure:

conf - 

  conf.py for setting and loging
  
  settings.toml - configuration settings data (path, hyperparameters, etc)
  
  
connector-

  connector.py - function for fetching data
  
model -

  conf - holds trained models
  
  your_model.py - main file for training models
  

util -

  util.py - functions to load/save models
  
  
entrypoint.py - 'face' of the project

To get predictions use prediction in entrypoint.

Entrypoint takes values for test (stored in settings toml under [PREDICT]:test) and choice of model (stored in settings toml under [MODEL]:top).

Specify rf or gb to train or use pre-trained random forest or gradient boosting

Training is started automaticaly if the model is not found 
