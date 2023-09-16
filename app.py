import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI # Brings in the OpenAI API
from PIL import Image
from langchain.agents import initialize_agent, AgentType, Tool # allows for agents to be created
from langchain.utilities import SerpAPIWrapper, PythonREPL # Search tool dependency




image = Image.open("StudyAngelLogo2.png")
llm = OpenAI(openai_api_key='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')# initializes the llm(the ai), so it's ready for use
openai_key = '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'



import os

SERPAPI_API_KEY = 'f4028f207d27e8125715e806749c114ba8d5c4efa5fea7cb19f64efb36bb518f'
os.environ['SERPAPI_API_KEY'] = SERPAPI_API_KEY

search = SerpAPIWrapper()
Search = Tool( #generates a tool for the agent to use
    func=search.run,
    name="Search",
    description="useful if you dont know the answer"
)
tools = [Search] # adds the search tool to the agent's toolbox

python_REPL = PythonREPL()
python_repl = Tool(# allows the AI to code
  name="python_repl",
  func=python_REPL.run,
  description="useful for when you need to use python to answer a question. You should input python code"
)
tools.append(python_repl)

agent = initialize_agent( # creates an agent with the tools
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, 
)



col1, col2 = st.columns(2)
with col1:
   st.title("\nStudyAngel Learning Assistant")
with col2:
   st.image(image)


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
        # number of questions
      with col2:
        topic = st.text_input("Topic: ")
        # topic of questions -- text input

      difficulty = st.slider("Difficulty", min_value=1, max_value=100, value=50)
      # difficulty of question

      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True
        pdf_question = "Generate " + str(number) + " questions about " + topic + "; [Difficulty Level: " + str(difficulty) + " out of 100]"

  elif option == "SmartReader":
    with st.form("smartReader"):
      topic = st.text_input("Topic: ")
      # topic from pdf -- text input
      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True
        pdf_question = "Condense the information about " + topic + " in a way that's easy to understand."
  
  else:
    with st.form("infoFinder"):
      pdf_question = st.text_input("Enter any question: ")
      # enter question about info from the pdf for the ai to answer
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
  # Practice Assistant
  option = st.selectbox("What would you like to do?", ("Generate by class", "Generate custom", "Solve"))
  gen = False # to trigger generation

  if option == "Generate by class":
    with st.form("byClass"):
      col1, col2, col3 = st.columns([0.5, 1, 1])
      with col1:
        number = st.number_input("Number of questions", min_value=1, max_value=5, value=1, step=1)
        # number of questions

      with col2:
        topic = st.text_input("Topic")
        # topic of questions

      with col3:
        curriculum = st.text_input("Class")
        # class curriculum

      difficulty = st.slider("Difficulty", min_value=1, max_value=100, value=50)
      # difficulty slider

      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True
        prompt = "Generate " + str(number) + " problems about " + topic + " from the " + curriculum + " curriculum without writing the solution(s); [Difficulty Level: " + str(difficulty) + "out of 100]"

  elif option == "Generate custom":
    with st.form("custom"):
      prompt = st.text_input("Enter any question prompt here:")
      # prompt for custom question generation
      submitted = st.form_submit_button("Generate")
      if submitted:
        gen = True

  else:
    with st.form("solve"):
      prompt = st.text_input("Enter any question here:")
      # prompt for question to be solved and explained
      submitted = st.form_submit_button("Generate")
      if submitted:
          gen = True



  if gen == True:
    response = llm(prompt=prompt) #When someone enters the prompt, a response is generated by the AI, which is then displayed in the website
    st.write(response)
    with st.expander("See explanation"):
      # put explanation behind an expander -- full solution + explanation hidden to the user without them opting to see it
      agentResponce = agent.run(response) # the agent is given the question, and tries to solve it
      explaination = llm("Explain how to find this answer: \"" + agentResponce + "\" to this question: " + response)
      st.write(explaination)