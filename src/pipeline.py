from src.utils.converter import Converter
from src.utils.question_type_predictor import QuestionTypePredictor

class Pipeline:
    def __init__(self):
        self.qtp = QuestionTypePredictor()
        self.conv = Converter()
        pass
    def launch(self, question) -> str:
        """
        Launches the Pipeline, gets results and returns Natural language response
        """
        response = ""
        # 1. predict question type
        qt = self.qtp.getQuestionType(question)
        if qt == "factual":
            # 2. Convert nl to sparql
            self.conv.factual_convert_nl_to_sparql(question)
            # 3. Result to sparql lookup
            # 4. Back to natural language
        elif qt == "embedding":
            # 2. Convert nl to sparql
            self.conv.embedded_convert_nl_to_sparql(question)
            # 3. Result to sparql lookup
            # 4. Back to natural language
        else:
            # hardened failsafe
            pass

        return response