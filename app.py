# -*- coding: utf-8 -*-
import os
import warnings
import textwrap
import streamlit as st

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from crewai import Agent, Task, Crew

# If needed, uncomment and run these installations before starting the app:
# !pip install langchain==0.3.10
# !pip install langchain-openai==0.2.11
# !pip install openai==1.55.1
# !pip install httpx==0.27.2
# !pip install jupyter_bokeh
# !pip install streamlit
# !pip install langchain-community
# !pip install pypdf
# !pip install unstructured
# !pip install pypandoc
# !pip install faiss-gpu
# !pip install chromadb
# !pip install langchain-cohere
# !pip install tiktoken
# !pip install crewai crewai_tools langchain_community

# If using langchain_community's PyPDFLoader:
from langchain_community.document_loaders import PyPDFLoader

warnings.filterwarnings('ignore')

##############################################################
# Set your OpenAI API Key here (please handle securely)
##############################################################
os.environ['OPENAI_API_KEY'] = "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"

def get_llm(temperature=0.7, model="gpt-4"):
    api_key = os.environ.get('OPENAI_API_KEY', None)
    if api_key is None:
        st.warning("Please set your OPENAI_API_KEY to proceed.")
        return None
    return ChatOpenAI(
        api_key=api_key,
        model_name=model,
        temperature=temperature
    )

##############################################################
# Title and Introduction
##############################################################

st.title("AI Drama Therapist")
st.markdown("""
This application simulates a drama therapy session using a Large Language Model.

- It collects the user's emotional background and demographic info.
- It assesses the user's psychological state.
- It decides on a suitable drama therapy play and character based on the user's situation.

Please follow the instructions in the chat below.
""")

##############################################################
# Load Documents and Create Vector Store
##############################################################

try:
    loader = PyPDFLoader('Play Database.pdf')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    llm_for_retrieval = get_llm(temperature=0, model="gpt-4")
    retriever = vector_store.as_retriever()
    retrieval_chain = RetrievalQA.from_chain_type(llm_for_retrieval, retriever=retriever)
except Exception as e:
    st.error("Error loading or processing 'Play Database.pdf'. Please ensure the file exists.")
    st.stop()

##############################################################
# Agents and Tasks (Crew AI)
##############################################################

analyst = Agent(
    role="Patient Analyst",
    goal="Analyze patient's condition",
    backstory=(
        "You help the therapist to evaluate the user's conditions. "
        "Based on {conversation}, you provide a summary of the situation and the user's feelings. "
        "Your work is the basis for the evaluator to rate the seriousness of the situation."
    ),
    allow_delegation=False,
    verbose=True
)

evaluator = Agent(
    role="Evaluator",
    goal="Evaluate seriousness",
    backstory=(
        "You are evaluating the user's psychological and emotional challenges. "
        "Your evaluation is based on the summary provided by Patient Analyst. "
        "You give a rating on a scale from 1 to 10. "
        "1 is mild, 10 is urgent. "
        "If >=9, get local help immediately."
    ),
    allow_delegation=False,
    verbose=True
)

therapist = Agent(
    role="General Therapist",
    goal="Provide advice",
    backstory=(
        "You are a therapist who gives advice based on demographics, conditions, "
        "mental health score and available time. "
        "Decide if drama therapy is needed. "
        "Criteria: If time ≤10 min: music/nature scene. "
        "If <30 min: relaxation techniques. "
        "If ≥30 min: full drama therapy."
    ),
    allow_delegation=False,
    verbose=True
)

drama_specialist = Agent(
    role="Drama Specialist",
    goal="Provide drama therapy",
    backstory=(
        "If drama therapy is decided, pick a suitable play from the 10 plays in {play_database}, "
        "choose a character for the patient to role-play based on demographics. "
        "If drama therapy is not chosen, wish them a good day."
    ),
    allow_delegation=False,
    verbose=True
)

summarize = Task(
    description=(
        "1. Draw important information from {conversation}. "
        "2. Summarize the user's basic info and the problem. "
        "Analyze the sentiment and tone of the user."
    ),
    expected_output="Two short paragraphs summarizing the user’s info and problem.",
    agent=analyst
)

evaluate = Task(
    description=(
        "Analyze {conversation} and the Analyst's report, provide a rating 1-10 "
        "for urgency of user's condition."
    ),
    expected_output="A rating from 1 to 10.",
    agent=evaluator
)

consult = Task(
    description=(
        "Based on the Analyst's summary and Evaluator's score, decide therapy. "
        "Use the time criteria for therapy type. Answer if drama therapy needed."
    ),
    expected_output="Yes/No drama therapy and instructions.",
    agent=therapist
)

play_task = Task(
    description=(
        "If drama therapy is needed, pick a play from {play_database} and a suitable character. "
        "If not needed, just send supportive message."
    ),
    expected_output="Name of play and character or supportive message.",
    agent=drama_specialist
)

crew = Crew(
    agents=[analyst, evaluator, therapist, drama_specialist],
    tasks=[summarize, evaluate, consult, play_task],
    verbose=True
)

##############################################################
# Conversation Setup
##############################################################

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    llm_for_chat = get_llm(temperature=0.7, model="gpt-4")
    memory = ConversationBufferMemory()
    st.session_state.chain = ConversationChain(llm=llm_for_chat, memory=memory)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

##############################################################
# Chat Input
##############################################################

user_input = st.chat_input("Hello there! How's your day been going?")
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from chain
    response = st.session_state.chain.run(user_input)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # If the user writes "done", we proceed with Crew analysis
    if "done" in user_input.lower():
        conversation_history_dict = {"user": [], "assistant": []}
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                conversation_history_dict["user"].append(msg["content"])
            elif msg["role"] == "assistant":
                conversation_history_dict["assistant"].append(msg["content"])

        inputs = {"conversation": conversation_history_dict, "play_database": chunks}

        recommendation = crew.kickoff(inputs)

        st.subheader("Therapy Recommendation")
        st.write(recommendation.raw)
