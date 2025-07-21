from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def create_classifier(name, params):
    if name == 'mlp':
        '''params={
            "hidden_layer_sizes": [512, 256],
            "early_stopping": True,
            "n_iter_no_change": 10,
            "max_iter": 300,
        }'''
        return MLPClassifier(**params)
    if name == 'xgboost':
        return xgb.XGBClassifier(**params)

