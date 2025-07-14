from sklearn.neural_network import MLPClassifier

def create_classifier(name, params):
    if name == 'mlp':
        return MLPClassifier(**params)
