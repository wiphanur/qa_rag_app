import json
import os
import streamlit as st
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
from copy import deepcopy
from tempfile import NamedTemporaryFile
    
def main():

    st.set_page_config(page_title = "Q&A application based on uploaded PDF using Llama Index & Llama2", page_icon="🦙")
    st.header('Q&A application based on uploaded PDF using Llama Index & Llama2"')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)

    embed_model = GradientEmbedding(
        gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
        gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
        gradient_model_slug="bge-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.chunk_size = 512

    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                accept_multiple_files = False, 
                                type=['pdf'])
        if st.button('Process'):
            with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
                f.write(docs.getbuffer())
                with st.spinner('Processing'):
                    documents = SimpleDirectoryReader(".").load_data()
                    index = VectorStoreIndex.from_documents(documents,
                                                            embed_model=embed_model)
                    query_engine = index.as_query_engine(llm=llm)
                    if "query_engine" not in st.session_state:
                        st.session_state.query_engine = query_engine
                    st.session_state.activate_chat = True

    if st.session_state.activate_chat:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})

            query_index_placeholder = st.session_state.query_engine
            pdf_response = query_index_placeholder.query(prompt)
            cleaned_response = pdf_response.response
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", 
                                              "avatar" :'🤖',
                                              "content": cleaned_response})
        else:
            st.markdown('Upload your PDFs to chat')


if __name__ == '__main__':
    main()