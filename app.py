import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI # Brings in the OpenAI API
from PIL import Image


image = Image.open("StudyAngelLogo2.png")
llm = OpenAI(openai_api_key='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')# initializes the llm(the ai), so it's ready for use
openai_key = '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'

col1, col2 = st.columns(2)
with col1:
   st.title("\nStudyAngel Learning Assistant")
with col2:
   st.image(image)

# PDF reader
pdf = st.file_uploader("Enter file here!", type = "pdf")
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    # Create the Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = openai_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

# PDF questions
pdf_question = st.text_input("Ask any question about this PDF: ")
if pdf_question:
    important_chunks = knowledge_base.similarity_search(pdf_question)
    chain = load_qa_chain(llm, chain_type = "stuff")
    pdfResponse = chain.run(input_documents = important_chunks, question = pdf_question)
    st.write(pdfResponse)





prompt = st.text_input("Enter any question here:")


if prompt:
    response = llm(prompt=prompt) #When someone enters the prompt, a responce is generated by the AI, which is then displayed in the website
    st.write(response)


