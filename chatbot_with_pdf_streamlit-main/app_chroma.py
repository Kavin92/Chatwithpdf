import streamlit as st
from streamlit_chat import message
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Configure the baseline configuration of the OpenAI library for Azure OpenAI Service.
OPENAI_API_KEY = "941f4e4fb3fe4a20ac7e2c9553e3a2c3"  # Your Azure OpenAI API key
OPENAI_API_BASE = "https://KAVINKUMARVS.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "gpt-35-turbo"  # Your deployment name for the chat model
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_EMBEDDING_MODEL_NAME"  # Replace with your embedding deployment name
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_TYPE = "azure"

# Set API configurations for Azure OpenAI (no need to import openai directly)
# Initialize session state to store user input and generated output.
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# Initialize embeddings and vector store using Azure
embed = OpenAIEmbeddings(
    deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME, 
    openai_api_key=OPENAI_API_KEY, 
    model=OPENAI_EMBEDDING_MODEL_NAME, 
    openai_api_type=OPENAI_API_TYPE, 
    chunk_size=1
)
db = Chroma(persist_directory="./chroma_db/", embedding_function=embed)

# Set web page title and icon.
st.set_page_config(
    page_title="Chatbot with PDF",
    page_icon=":robot:"
)

# Set web page title and markdown.
st.title('Chatbot with PDF')
st.markdown(
    """
    This is the demonstration of a chatbot with PDF with Azure OpenAI, Chroma, and Streamlit.
    I read the book Machine Learning Yearning by Andrew Ng. Please ask me any questions about this book.
    """
)

# Define a function to get user input.
def get_input_text():
    input_text = st.text_input("You: ", "Hello!", key="input")
    return input_text 

# Define a function to inquire about the data in the vector store.
def query(payload, docs, chain):
    response = chain.run(input_documents=docs, question=payload)
    return {"generated_text": response}

user_input = get_input_text()

# Perform similarity search using the vector store.
docs = db.similarity_search(user_input)

# Initialize the Azure OpenAI ChatGPT model.
llm = AzureChatOpenAI(
    deployment_name=OPENAI_DEPLOYMENT_NAME, 
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE, 
    openai_api_version=OPENAI_API_VERSION, 
    openai_api_type=OPENAI_API_TYPE, 
    temperature=0
)

# Initialize the question answering chain.
chain = load_qa_chain(llm, chain_type="stuff")

# Generate the chatbot response if user input is provided.
if user_input:
    output = query(
        {
            "inputs": {
                "past_user_inputs": st.session_state.past,
                "generated_responses": st.session_state.generated,
                "text": user_input,
            },
            "parameters": {"repetition_penalty": 1.33}
        },
        docs=docs,
        chain=chain
    )
    
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])

# Display the conversation history in the Streamlit app.
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
