# -*- coding: utf-8 -*-
import os
import warnings
import streamlit as st

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from crewai import Agent, Task, Crew

warnings.filterwarnings('ignore')

##############################################################
# Retrieve the OpenAI API Key from environment variable
##############################################################
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # If the key is not found, display a warning message and stop the app.
    st.error("OPENAI_API_KEY not found. Please set it as an environment variable on your EC2 instance or securely retrieve it.")
    st.stop()

def get_llm(temperature=0.7, model="gpt-4"):
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
- It decides if drama therapy is needed based on the user's situation.

Type "done" when you have provided enough context, and the system will provide a recommendation.
""")

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
        "If drama therapy is decided, pick a suitable play from {play_database}, "
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
        "Use the time criteria for therapy type. Answer if drama therapy is needed."
    ),
    expected_output="Yes/No drama therapy and instructions.",
    agent=therapist
)

play_task = Task(
    description=(
        "If drama therapy is needed, pick a play from {play_database} and a suitable character. "
        "If not needed, just send a supportive message."
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
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory()
    st.session_state.chain = ConversationChain(llm=llm_for_chat, memory=memory)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

##############################################################
# Placeholder play_database (no external files)
##############################################################
play_database = [
    "The Overcoming Journey",
    "Shadows of the Mind",
    "Sunrise at Dawn",
    "Lost and Found",
    "Heart's Echo",
    "Steps of Courage",
    "Waves of Emotion",
    "The Silent Voice",
    "Reflections on Glass",
    "A Path Untold"
]

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

        inputs = {"conversation": conversation_history_dict, "play_database": play_database}

        recommendation = crew.kickoff(inputs)

        st.subheader("Therapy Recommendation")
        st.write(recommendation.raw)
