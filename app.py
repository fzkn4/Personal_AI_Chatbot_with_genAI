from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain import FAISS
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain




#loading environment variables 
load_dotenv()
#creating model
llm = GoogleGenerativeAI(
    model='gemini-1.5-pro',
    temperature = 0.5,
    streaming=True 
    )
#creating tools, duckgo search engine 
tools = load_tools(
    ['ddg-search']
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

#creating instance of GoogleGenerativeAIEmbeddings
embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')



#chatgpt-like text prompt
def text_prompt():
    if prompt := st.chat_input():
        st.chat_message('user').write(prompt)
    
        if prompt == 'exit':
            st.stop()
        with st.chat_message('assistant'):
            st.write('Im thinking...')
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(prompt, callbacks=[st_callback])
            st.write(response)

#chat with PDF
def process_file():
    st.header("Chat with PDF 💬")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    #st.write(pdf)
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # it will divide the text into 800 chunk size each (800 tokens)
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(text=text)
        
        
        # st.write(chunks[1])
        

        knowledge_base  = FAISS.from_texts(chunks, embedding)
            
        
        # Accept user questions/query
        query = st.text_input("Ask your questions about your PDF file")
        #st.write(query)
        
        if query:
            docs = knowledge_base.similarity_search(query)

            llm = GoogleGenerativeAI(
            model='gemini-1.5-pro',
            temperature = 0.5,
            streaming=True 
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            
            st.success(response)


def main():
    #side bar contents
    with st.sidebar:
        #Webpage title 
        st.title('PERSONAL AI CHATBOT WITH LANGCHAIN')
        st.markdown("""
            *`Your personal AI Chatbot`*
                """)

        add_selectbox = st.selectbox(
                                "Options: ",
                                    ("Text", "Audio", "Document")
                                )

    with st.chat_message('Assistant'):
        st.write('Panagdait! How can I help you today? 🤗')


    if add_selectbox == 'Text':
        text_prompt()
    elif add_selectbox == 'Document':
        process_file()
    elif add_selectbox == 'Audio':
        pass


if __name__ == '__main__':
    main()