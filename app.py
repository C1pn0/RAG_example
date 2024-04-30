from langchain_pinecone import PineconeVectorStore
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import spacy
import gradio as gr


# Setup for language processing model and dependencies
try:
    spacy.cli.download("en_core_web_sm")  # Download Spacy's English model
    os.system('pip3 install unstructured[pdf]')  # Install additional package for handling PDFs
except Exception as e:
    pass  # Ignore exceptions that may occur during setup

# Configuration for using ChatGPT with OpenAI API
llm = ChatOpenAI(model_name='gpt-4', max_tokens=488,
                 model_kwargs={
                     "stop": ["\nQ:", "\nA:"],  # Define stopping criteria for model responses
                 })

# Loader configurations for different file types
loaders = {
    '.csv': DirectoryLoader(path="flattened_data", glob="**/*_all.csv")
    # Additional file types and their loaders can be added here
}

# Set up a vector store for retrieval-based question answering using Pinecone
vectorstore = PineconeVectorStore(index_name='rag-example',
                                  embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
                                  text_key='abstract')

# Function to generate prompt templates for the language model
def get_prompt(instruction, examples, new_system_prompt):
    """
    Function to generate prompt templates for the language model.
    Args:
        instruction (str): Instruction for the prompt.
        examples (str): Examples for the prompt.
        new_system_prompt (str): New system prompt.
    Returns:
        str: The generated prompt template.
    """
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  SYSTEM_PROMPT + instruction  + "\n" + examples
    return prompt_template

# Define prompt boundaries
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Define system prompt with guidelines for generating responses
sys_prompt = """\
You are a helpful, respectful and honest assistant designed to assist with. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Set up instruction and example format for the prompt
instruction = """CONTEXT:/n/n {context}/n
"""

examples = """
Q: {question}
A: """


# Configure the QA chain with the customized prompt template
template = get_prompt(instruction, examples, sys_prompt)
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# Define a function to handle chat responses
def chat_response(msg, history=None):
    """
    Function to handle chat responses.
    Args:
        msg (str): The message to respond to.
        history (str): The chat history.
    Returns:
        str: The chat response.
    """
    if history:
        vectorstore.embeddings.embed_documents(OpenAIEmbeddings(model='text-embedding-ada-002'), texts=history[-1], namespace='history')
    return qa_chain({"query": msg})["result"]

# Setup a Gradio interface for the application
demo = gr.ChatInterface(chat_response)

# Launch the application
if __name__ == "main":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inline", help="Inline the chat response for jupyter notebook", action="store_true")
    if parser.parse_args().inline:
        demo.launch(inline=True)
    else:
        demo.launch()
