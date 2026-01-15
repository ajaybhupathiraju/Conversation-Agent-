# Conversation-Agent- (Option 1: Chat With Your Docs)
Conversation-Agent is runs on LangGraph-driven agentic workflow, this agent utilizes a stateful, graph-based architecture to reason through complex user queries and provide contextually aware responses.

## Table of Contents
- [Quick setup instructions](#quick-setup-instructions)
- [Architecture Overview](#architecture-overview)
- [Productionize solution](#productionize-solution)
- [Tech Stack Used](#tech-stack-used)
- [Results](#results)


## Quick setup instructions

    - create venv -> uv add venv
    
    - activate venv -> .venv\scripts\activate
    
    - uv sync -> install dependencies
    
    - run python PlanningAgent.py

    Note : .env file is required to run this project. It has been excluded for security reasons.
    
## Architecture Overview
The Sequential Design Pattern in LangGraph connects AI agents in a sequence order, One agent finishes its specific task and passes its work directly to the next agent. 
This is used for simple tasks that must happen in a specific, step-by-step order.

<img width="107" height="333" alt="image" src="https://github.com/user-attachments/assets/19dc95fc-3943-4dc0-aa2f-cb9202fd7125" />

**_Planning Agent_** : The first node in the graph receives the user's input such as the study, drug, and query details and saves it directly into the graph's state.

**_Safety Agent_** : The second node in the graph, which processes the output from the Planning Agent. It performs two key functions:

1. It generates embeddings for the user input to retrieve and aggregate relevant records from the RAG system as context.

2. It sends the user query and retrieved context to the LLM with specific instructions to generate a response.


## productionize solution





## Tech Stack Used

Modules               |                    | 
--------------------- | ------------------ | 
Programming language  |   Python           |  
LLM provider          |   Azure Open AI    |  
Vector database       |   Qdrant           |  
Orchestration Frwk    |   RAG and Langraph |   
Architecture          |   Agentic Sequence | 


## Results

Input :

<img width="695" height="100" alt="image" src="https://github.com/user-attachments/assets/09198932-3256-4bcb-beab-c7b0aff27665" />

Agent Response :

<img width="951" height="236" alt="image" src="https://github.com/user-attachments/assets/38fa93a6-fd34-4c42-aaee-7282e60b8c0d" />
