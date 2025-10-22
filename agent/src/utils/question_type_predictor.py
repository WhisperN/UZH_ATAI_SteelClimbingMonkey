from src.nl.phi3mini128k import Phi3Mini128k
import re

class QuestionTypePredictor:
    def __init__(self):
        self.phi_model = Phi3Mini128k()

    def getQuestionType(self, question):
        """
        Predicts the question type according to these classes:
        - factual
        - embedding
        """
        classified = self.phi_model.ask(question)
        match = re.search(r"\b(Factual|Embedding)\b", classified, re.IGNORECASE)
        if match:
            classified = match.group(1).capitalize()
        else:
            classified = "Factual"
        return classified