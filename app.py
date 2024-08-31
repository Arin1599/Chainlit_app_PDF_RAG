import os
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
import chainlit as cl

# Uncomment and adjust this section if you are using environment variables
#from dotenv import load_dotenv
#load_dotenv()  
#groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM models
llm_local = ChatOllama()
llm_groq = ChatGroq(
    # Uncomment and adjust this if you need to use an API key or specific model
    #groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768' 
)

@cl.on_chat_start
async def on_chat_start():
    files = None  # Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]  # Get the first uploaded file

    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
    pdf_text = ""
    try:
        pdf = PyPDF2.PdfReader(file.path)
        for page in pdf.pages:
            pdf_text += page.extract_text()
    except Exception as e:
        await cl.Message(content=f"Failed to process PDF: {str(e)}").send()
        return

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store with embeddings
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text") #In order to ember nomic, youhave to download ollam on your local machine and run the following command in your terminal : ollama pull nomic-embed-text
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas
        )
    except Exception as e:
        await cl.Message(content=f"Failed to create embeddings or Chroma vector store: {str(e)}").send()
        return

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    # Store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="Chain not found. Please restart the chat session.").send()
        return

    # Callbacks happen asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    try:
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]
    except Exception as e:
        await cl.Message(content=f"Failed to process your request: {str(e)}").send()
        return

    text_elements = []  # Initialize list to store text elements
    source_links = []  # Initialize list to store source links


    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    # Return the results
    await cl.Message(content=answer).send()
