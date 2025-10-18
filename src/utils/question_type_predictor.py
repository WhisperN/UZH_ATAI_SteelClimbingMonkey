from src.nl.phi3mini128k import Phi3Mini128k

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
        # Optional string parsing
        print(classified)
        return classified