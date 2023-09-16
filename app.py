import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI # Brings in the OpenAI API

from PIL import Image
from langchain.agents import initialize_agent, AgentType, Tool # allows for agents to be created
from langchain.utilities import SerpAPIWrapper # Search tool dependency
from pydantic import BaseModel, Field
from langchain.chains import LLMMathChain




image = Image.open("StudyAngelLogo2.png")
llm = OpenAI(openai_api_key='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')# initializes the llm(the ai), so it's ready for use
openai_key = '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'


col1, col2 = st.columns(2)
with col1:
   st.title("\nStudyAngel Learning Assistant")
with col2:
   st.image(image)

import os

SERPAPI_API_KEY = '7778082c548d3cea65fe82dfdcb87e7ddb8ad8d7da3ae08056b49a0ce29bf83c'
os.environ['SERPAPI_API_KEY'] = SERPAPI_API_KEY

search = SerpAPIWrapper()
Search = Tool( #generates a tool for the agent to use
    func=search.run,
    name="Search",
    description="useful if you dont know the answer"
)
tools = [Search] # adds the search tool to the agent's toolbox



llm_math_chain = LLMMathChain(llm=llm, verbose=True) # initializes the math chain AI
class CalculatorInput(BaseModel):
    question: str = Field()
Calculator = Tool( # Implements the calculator as a tool available to the Agent
    func=llm_math_chain.run,
    name="Calculator",
    description="useful for when you need to answer questions about math",
)
tools.append(Calculator) # adds the calculator tool to the agent's toolbox





agent = initialize_agent( # creates an agent with the tools
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, 
)


tab1, tab2 = st.tabs(["Practice Assistant", "PDF Reader"])

with tab2:
  # PDF reader
  pdf = st.file_uploader("Enter file here!", type = "pdf")
  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
      # Split text into chunks
      text_splitter = CharacterTextSplitter(# PDF reader
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
      )
      chunks = text_splitter.split_text(text)
      # Create the Embeddings
      embeddings = OpenAIEmbeddings(openai_api_key = openai_key)
      knowledge_base = FAISS.from_texts(chunks, embeddings)

  # options: generate, smartreader, info finder
  option = st.selectbox("What would you like to do?", ("Generate question(s)", "SmartReader", "InfoFinder"))
  gen = False

  if option == "Generate question(s)":
    with st.form("genQues"):
      col1, col2 = st.columns([0.4, 1])
      with col1:
        number = st.number_input("Number of questions", min_value=1, max_value=5, value=1, step=1)
      with col2:
        topic = st.text_input("Topic: ")

      difficulty = st.slider("Difficulty", min_value=1, max_value=100, value=50)

      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True
        pdf_question = "Generate " + str(number) + " questions about " + topic + "; [Difficulty Level: " + str(difficulty) + " out of 100]"

  elif option == "SmartReader":
    with st.form("smartReader"):
      topic = st.text_input("Topic: ")
      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True
        pdf_question = "Condense the information about " + topic + " in a way that's easy to understand."
  
  else:
    with st.form("infoFinder"):
      pdf_question = st.text_input("Enter any question: ")
      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True

  # Generate the text based on whatever input the user gives
  if gen == True:
      important_chunks = knowledge_base.similarity_search(pdf_question)
      chain = load_qa_chain(llm, chain_type = "stuff")
      pdfResponse = chain.run(input_documents = important_chunks, question = pdf_question)
      st.write(pdfResponse)

    



with tab1:

  option = st.selectbox("What would you like to do?", ("Generate by class", "Generate custom", "Solve"))
  gen = False

  if option == "Generate by class":
    with st.form("byClass"):
      col1, col2, col3 = st.columns([0.5, 1, 1])
      with col1:
        number = st.number_input("Number of questions", min_value=1, max_value=5, value=1, step=1)
      with col2:
        topic = st.text_input("Topic")

      with col3:
        curriculum = st.text_input("Class")

      # difficulty slider
      difficulty = st.slider("Difficulty", min_value=1, max_value=100, value=50)

      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True
        prompt = "Generate " + str(number) + " problems about " + topic + " from the " + curriculum + " curriculum without writing the solution(s); [Difficulty Level: " + str(difficulty) + "out of 100]"

  elif option == "Generate custom":
    with st.form("custom"):
      prompt = st.text_input("Enter any question here:")
      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True

  else:
    with st.form("solve"):
      prompt = st.text_input("Enter any question here:")
      submitted = st.form_submit_button("Generate")
      if submitted:
          gen = True



  if gen == True:
    response = llm(prompt=prompt) #When someone enters the prompt, a response is generated by the AI, which is then displayed in the website
    st.write(response)
    with st.expander("See explanation"):
      agentResponce = agent.run(response) # the agent is given the question, and tries to solve it
      explaination = llm("Explain how to find this answer: \"" + agentResponce + "\" to this question: " + response)
      st.write(explaination)