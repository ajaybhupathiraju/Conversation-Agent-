from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AnyMessage,AIMessage
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from typing_extensions import Annotated,Literal
from langgraph.graph.message import add_messages
from azure.identity import ClientSecretCredential
from SafetyAgent import SafetyAgent

import logging
from IPython.display import Image, display
logging.basicConfig(
    level=logging.INFO, # Set the global threshold to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

class PlanningAgent:
    """This class is used to create a planning agent."""
    def __init__(self):
        load_dotenv()
        self.safety_agent = SafetyAgent()
    
    class State(TypedDict):
        response: str
        study : str
        drug : str
        user_query: str

    def safety_node(self,state:State)-> State:
        """This node is used to get the safety summary."""
        try:
            response = self.safety_agent.agent_response(study_name=state['study'], drug_name=state['drug'], user_query=state['user_query'])
            return {
                'response': response,
            }
        except Exception as e:
            logger.error(f"Error in safety node: {e}")
    
    def planning_node(self, state:State) -> State:
        """This function is used to ask the user inputs like study name and drug name and query text"""
        try:
            study = input("Enter the study name: ")
            drug = input("Enter the drug name: ")
            user_query = input("Enter the your query: ")
            return {"study": study, "drug": drug, "user_query": user_query}
        except Exception as e:
            logger.error(f"Error in planning_node: {e}")

    def build_graph(self) -> StateGraph:
        """This function is used to build the graph for the planning agent"""
        try:
            graph = StateGraph(self.State)
            graph.add_node("safety", self.safety_node)
            graph.add_node("planning", self.planning_node)
            graph.add_edge(START, "planning")
            graph.add_edge("planning", "safety")
            graph.add_edge("safety", END)
            return graph.compile()
        except Exception as e:
            logger.error(f"Error in build_graph: {e}")
            return e

    def run(self):
        """This function is used to run the planning agent"""
        logger.info("PlanningAgent -> run invoked")
        try:
            graph = self.build_graph()
            png_data = graph.get_graph().draw_mermaid_png()
            file_path = "graph.png"
            with open(file_path, "wb") as f:
                f.write(png_data)

            final_state = graph.invoke({"response": "","study": "", "drug": "", "user_query": ""})
            return final_state
        except Exception as e:
            logger.error(f"Error in run: {e}")
            return e

if __name__ == "__main__":
    planning_agent = PlanningAgent()
    final_state = planning_agent.run()
    logger.info("******* Answer from Agent *******")
    logger.info(final_state['response'])