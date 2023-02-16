from src.services.metier.classificationMetier import ClassificationMetier

class ClassificationApp:
    _metier = None

    def __init__(self, metier = ClassificationMetier()) -> None:
        if metier:
            self._metier = metier

    def classify(self):
        return self._metier.classify()
