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
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

        self.g = Graph().parse("https://files.ifi.uzh.ch/ddis/teaching/2025/ATAI/dataset/graph.nt")
        '''print(f"Graph has {len(self.g)} triples.\n")

        for i, (s, p, o) in enumerate(self.g):
            print(s, p, o)'''

        print('parsing done')
        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

        self.pipeline = Pipeline()

    def listen(self):
        """Start listening for events."""
        self.speakeasy.start_listening()

    def on_new_message(self, message : str, room : Chatroom):
        """Callback function to handle new messages."""
        print(f"New message in room {room.room_id}: {message}")
        self.__execute_sparql(message, room)

    def on_new_reaction(self, reaction : str, message_ordinal : int, room : Chatroom):
        """Callback function to handle new reactions."""
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        # Implement your agent logic here, e.g., respond to the reaction.
        room.post_messages(f"Thanks for your reaction: '{reaction}'")


    def __execute_sparql(self, message: str, room: Chatroom):
        """Execute a SPARQL query after extracting the actual SPARQL query."""
        try:
            query = self.pipeline.getQuery(message)
            match = self.__extract_messages(query)
            if match:
                query = match.group(1)
                literals = self.__extract_literals(query)
                processed_response = self.pipeline.getResponse(literals)
                room.post_messages(str.join(', ', processed_response))
            else:
                literals = self.__extract_literals(message)
                if literals:
                    room.post_messages(', '.join(map(str, literals)))
                else:
                    room.post_messages("There was no valid SPARQL query in your prompt, please encode the SPARQL query within single quotes or only provide the SPARQL query.")
        except Exception:
            print(traceback.format_exc())
            room.post_messages("There was no valid SPARQL query in your prompt, please encode the SPARQL query within single quotes or only provide the SPARQL query.")

    def __extract_literals(self, query: str):
        """Extract literals from the SPARQL query."""
        literals = []
        if len(query):
            for row in self.g.query(query):
                for value in row:
                    if isinstance(value, Literal):
                        literals.append(value.toPython())
        return literals

    def __extract_messages(self, message: str):
        """Extract messages from the SPARQL query."""
        pattern = r"'''(.*?)'''"
        return re.search(pattern, message, re.DOTALL)


    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("SteelClimbingMonkey", "Yd0Gg8nK")
    demo_bot.listen()