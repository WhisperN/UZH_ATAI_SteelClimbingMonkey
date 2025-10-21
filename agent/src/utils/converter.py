from src.nl.llama3b import Llama3B

class Converter:
    def __init__(self):
        self.llama3b = Llama3B()

    def factual_convert_nl_to_sparql(self, nl_input) -> str:
        """
        Takes natural language inputs and returns a sparql query
        Speciality: Factual questions
        """
        query = self.llama3b.ask(nl_input)
        return query

    def embedded_convert_nl_to_sparql(self, nl_input):
        """
        Takes natural language inputs and returns a sparql query
        Speciality: Embedded questions
        """
        query = self.llama3b.ask(nl_input)
        return query

    def query_result_to_nl(self, result):
        """
        Accepts a list of result entities and generates a natural
        language response
        """
        query = self.llama3b.ask(result)
        return query

