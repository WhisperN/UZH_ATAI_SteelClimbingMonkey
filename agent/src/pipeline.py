from agent.src.utils.converter import Converter
from agent.src.utils.question_type_predictor import QuestionTypePredictor

class Pipeline:
    def __init__(self):
        self.qtp = QuestionTypePredictor()
        self.conv = Converter()

    def getQuery(self, question) -> str:
        """
        Launches the Pipeline, gets results and returns SPARQL
        """
        query = ""
        # 1. predict question type
        qt = self.qtp.getQuestionType(question)
        if qt == "Factual":
            # 2. Convert nl to sparql
            query = self.conv.factual_convert_nl_to_sparql(question)
            # 3. Result to sparql lookup
        elif qt == "Embedding":
            # 2. Convert nl to sparql
            query = self.conv.embedded_convert_nl_to_sparql(question)
            # 3. Result to sparql lookup
            # lookup = "query"
            # 4. Back to natural language
            # response = self.conv.query_result_to_nl(lookup)
            # 5. Send to agent
        else:
            # hardened failsafe
            pass
        print(query)
        return query

    def getResponse(self, query_result) -> str:
        return self.conv.query_result_to_nl(query_result)