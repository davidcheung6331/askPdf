from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os
import streamlit as st

from langchain.agents import initialize_agent
from langchain.agents import AgentType

from PIL import Image
image = Image.open("searchpdf.png")


st.set_page_config(
    page_title="Ask  PDFs",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "Demo Page by AdCreativeDEv"
    }
)
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.image(image, caption='created by MJ')

# Pydantic is a Python library for data modeling/parsing 
# declare a DocumentInput class , which inherits from the BaseModel.
# represents the input schema for the PDFdocuments.
class DocumentInput(BaseModel):
    question: str = Field()


# Create  an empty list for tools 
tools = []
files = [
    
    {
        "name": "job1", 
        "path": "job1.pdf",
    }, 
    
    {
        "name": "job2", 
        "path": "job2.pdf"
    }
]
    


st.title("📔📔 Ask your PDFs")

system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613", 
)

#  initializes the agent, tools , the language model instance
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
)



st.subheader("Step 1. Load PDF documents ?")
st.caption("It will Load sample Job Posts (PDF files)")
if st.button("Upload"):
    # Looping over the files list , 2 files
    for file in files:
        # For each file, Loading and splitting the PDF document:
        filename = file["path"] 
        loader = PyPDFLoader(file["path"])
        print(f"1. Loading pdf :  {filename}")
        st.caption("1. Loading pdf : " + filename)

        # creates a PyPDFLoader by filename and then loads and splits the PDF into individual pages.
        pages = loader.load_and_split()
        # initializes a CharacterTextSplitter with a chunk size of 1000 characters and no overlap. 
        # It then splits the text into chunks of the specified size.
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        # created to generate embeddings for the text chunks
        st.caption("2. create embeddings for chunks")
        embeddings = OpenAIEmbeddings()
        # a retriever is created using FAISS with the document chunks and embeddings.
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()
        # creates a Tool instance with DocumentInput schema, the file name, a description, and the retrieval 
        print("creates a Tool instance with DocumentInput schema, the file name, a description, and  retrieval ")
        st.caption("3. store embeddings in vector db")
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=file["name"], 
                description=f"useful when you want to answer questions about {file['name']}",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            )
        )



# calls the agent with a dictionary containing the input question. 
# agent will process the question and return a response.
st.subheader("Step 2 : After upload, enter the Prompt for Search" )
QueryString = "which job need the experience in charity ?"
QueryString = st.text_input('Sample Prompt ie : ', QueryString)
finalQuerySubmit = "input : " + QueryString
st.text("Final Prompt Submit to Agnet :" + finalQuerySubmit )
if st.button("Search"):
    result = agent({finalQuerySubmit})
    st.info(result)
    # subprocess.call(["say", result])
    print(f"Analysis result: {result}")

#result = agent({"input": "which job need the experience in charity ?"})
#st.info(result )






















log = """
> Entering new AgentExecutor chain...
I need to find out which job requirement accepts a high diploma.
Action: job1

Action Input: 'high diploma'

Observation: {'query': "'high diploma'", 'result': "A high diploma refers to a diploma obtained after completing a high school education. It is a level of education that is below a bachelor's degree but above a secondary school certificate."}

Thought:I have obtained information about what a high diploma is.

Action: job1

Action Input: 'job requirement

Observation: {'query': "'job requirement'", 'result': "The job requirements for this position include:\n\n- A bachelor's degree, associate degree, or high diploma in any discipline\n- Strong verbal and communication skills\n- Strong organizational skills and attention to detail\n- Proficiency in MS Office Applications and Google Drive functions, Photoshop, and Canva\n- A minimum of 1 year relevant experience\n- Willingness to work on weekends when necessary (off-days will be given in-lieu)"}
I have obtained information about the job requirements for job1.

Action: job1

Action Input: 'accept high diploma'

Observation: {'query': "'accept high diploma'", 'result': 'Yes, the company accepts applicants with a high diploma as long as they meet the other requirements mentioned in the job description.'}
I now know the final answer.

Final Answer: Yes, job1 accepts applicants with a high diploma as long as they meet the other requirements mentioned in the job description.

> Finished chain.
Analysis result: {'input': {'input : which job requirement accept high diploma ?'}, 'output': 'Yes, job1 accepts applicants with a high diploma as long as they meet the other requirements mentioned in the job description.'}


########

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Create  an empty list for tools 
tools = []
files = [
    
    {
        "name": "job1", 
        "path": "job1.pdf",
    }, 
    
    {
        "name": "job2", 
        "path": "job2.pdf"
    }
]
for file in files:
    filename = file["path"] 
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"], 
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        )
    )
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613", 
)
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
)

QueryString = "which job requirement accept high diploma ?"
QueryString = st.text_input('input Prompt below, you can amend it : ',QueryString)
st.text("Final Prompt Submit to Agnet :" + finalQuerySubmit )
result = agent({finalQuerySubmit})



"""

with st.expander("explanation"):
    
    st.code(log)