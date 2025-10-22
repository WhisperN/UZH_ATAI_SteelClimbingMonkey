import re
import time
from rdflib import Literal
import traceback


from rdflib import Graph

from speakeasypy import Chatroom, EventType, Speakeasy

from src.pipeline import Pipeline

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'

class Agent:
    def __init__(self, username, password):
        self.username = username
        print('Loading graph and initialising pipeline')
        g = Graph().parse("https://files.ifi.uzh.ch/ddis/teaching/2025/ATAI/dataset/graph.nt")
        g.bind("wd", "http://www.wikidata.org/entity/")
        g.bind("wdt", "http://www.wikidata.org/prop/direct/")
        g.bind("p", "http://www.wikidata.org/prop/")
        g.bind("ps", "http://www.wikidata.org/prop/statement/")
        g.bind("pq", "http://www.wikidata.org/prop/qualifier/")
        g.bind("schema", "http://schema.org/")
        g.bind("ddis", "http://ddis.ch/atai/")
        self.g = g
        print('Graph loaded')

        self.pipeline = Pipeline()

        '''print(f"Graph has {len(self.g)} triples.\n")

        for i, (s, p, o) in enumerate(self.g):
            print(s, p, o)'''

        print('parsing done')
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        """Start listening for events."""
        self.speakeasy.start_listening()

    def on_new_message(self, message : str, room : Chatroom):
        """Callback function to handle new messages."""
        try:

            print(f"New message in room {room.room_id}: {message}")
            self.__execute_sparql(message, room)
        except Exception:
            print(traceback.format_exc())
            room.post_messages("Sadly, I am too tired to answer that, ask another time.")

    def on_new_reaction(self, reaction : str, message_ordinal : int, room : Chatroom):
        """Callback function to handle new reactions."""
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        # Implement your agent logic here, e.g., respond to the reaction.
        room.post_messages(f"Thanks for your reaction: '{reaction}'")


    def __execute_sparql(self, message: str, room: Chatroom):
        """Execute a SPARQL query after extracting the actual SPARQL query."""
        query = self.pipeline.getQuery(message)
        all_results = []

        print("query: ", query)
        literals = self.__extract_literals(query)
        print("literals: ", literals)
        pr = ""
        if literals:
            pr = self.pipeline.getResponse(message, literals)
            print("pr: ", pr)
        additional_info = ""
        actual_response = self.pipeline.getResponse(message, pr, additional_info)
        print("actual_response: ", actual_response)
        room.post_messages(actual_response)


    def __extract_literals(self, query: str):
        literals = []
        print(query)
        if len(query):
            for row in self.g.query(query):
                for value in row:
                    if isinstance(value, Literal):
                        literals.append(value.toPython())
        return literals

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("SteelClimbingMonkey", "Yd0Gg8nK")
    demo_bot.listen()