from src.nl.llama3b import Llama3B
from src.nl.occiglot import OcciglotSPARQL

class Converter:
    def __init__(self):
        self.occi = OcciglotSPARQL()
        self.llama3b = Llama3B()

    def factual_convert_nl_to_sparql(self, nl_input):
        """
        Takes natural language inputs and returns a sparql query
        Speciality: Factual questions
        """
        query, entities = self.occi.ask(nl_input)
        return query, entities

    def plaubalise(self, question, result, entities):
        """
        Accepts a list of result entities and generates a natural
        language response
        """
        query = self.llama3b.askToPlaubalise(question, result, entities)
        return query

    def embedding_fallback(self, question):
        """
        Is used to fallback to embeddings if the message could not be translated to sparql directly or if the response was not sensible
        """
        return self.occi.embedding_fallback(question)

    def query_result_to_nl(self, question, result, additional_info):
        """
        Accepts a list of result entities and generates a natural
        language response
        """
        query = self.llama3b.ask(question, result, additional_info)
        return query