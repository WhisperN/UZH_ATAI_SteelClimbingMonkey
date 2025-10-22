from src.utils.converter import Converter
from src.utils.question_type_predictor import QuestionTypePredictor

class Pipeline:
    def __init__(self):
        self.conv = Converter()

    def getQueries(self, question):
        """
        Launches the Pipeline, gets results and returns SPARQL
        """
        # 1. predict question type
        # 2. Convert nl to sparql
        # 3. Result to sparql lookup
        # 2. Convert nl to sparql
        # 3. Result to sparql lookup
        # lookup = "query"
        # 4. Back to natural language
        # response = self.conv.query_result_to_nl(lookup)
        # 5. Send to agent
        # hardened failsafe
        queries, entities = self.conv.factual_convert_nl_to_sparql(question)
        return queries, entities

    def getResponse(self, question, query_result, additional_info = "") -> str:
        return self.conv.query_result_to_nl(question, query_result, additional_info)
    def plaubalise(self, question, literals, entities):
        return self.conv.plaubalise(question, literals, entities)
    def embedding_fallback(self, question):
        return self.conv.embedding_fallback(question)
