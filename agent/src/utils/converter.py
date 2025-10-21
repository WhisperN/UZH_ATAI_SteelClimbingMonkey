from src.nl.qwen257B import Qwen257B

class Converter:
    def __init__(self):
        self.qwen257B = Qwen257B()

    def factual_convert_nl_to_sparql(self, nl_input) -> str:
        """
        Takes natural language inputs and returns a sparql query
        Speciality: Factual questions
        """
        query = self.qwen257B.ask(nl_input)
        return query

    def embedded_convert_nl_to_sparql(self, nl_input):
        """
        Takes natural language inputs and returns a sparql query
        Speciality: Embedded questions
        """
        query = self.qwen257B.ask(nl_input)
        return query

    def query_result_to_nl(self, result):
        """
        Accepts a list of result entities and generates a natural
        language response
        """
        query = self.qwen257B.ask(result)
        return query

