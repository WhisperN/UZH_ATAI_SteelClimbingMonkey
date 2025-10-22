from src.nl.llama3b import Llama3B
from src.nl.occiglot import OcciglotSPARQL

class Converter:
    def __init__(self):
        self.occi = OcciglotSPARQL()
        self.llama3b = Llama3B()

    def factual_convert_nl_to_sparql(self, nl_input) -> str:
        """
        Takes natural language inputs and returns a sparql query
        Speciality: Factual questions
        """
        query = self.occi.ask(nl_input)
        return query

    def query_result_to_nl(self, question, result, additional_info):
        """
        Accepts a list of result entities and generates a natural
        language response
        """
        query = self.llama3b.ask(question, result, additional_info)
        return query

