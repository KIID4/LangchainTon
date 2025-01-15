import os
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# 오픈AI API 키 설정
os.environ["OPENAI_API_KEY"] = ''

# Streamlit UI 및 기존 코드 유지
# PDF File Load
@st.cache_resource
def load_all_pdfs(folder_path):
    all_pages = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path) 
            pages = loader.load_and_split()
            all_pages.extend(pages)  # 모든 PDF 파일의 페이지들을 합침
            
    return all_pages


# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    
    vectorstore = Chroma.from_documents(
        split_docs, 
        OpenAIEmbeddings(model ='text-embedding-3-small'),
        persist_directory = persist_directory
    )
    
    return vectorstore


# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        
        return Chroma(
            persist_directory = persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
        )
    else:
        return create_vector_store(_docs)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize the LangChain components
@st.cache_resource
def chaining():
    folder_path = r"../data/gathering"  # data 폴더 경로
    pages = load_all_pdfs(folder_path)  # 모든 PDF 파일 로드
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # Define the answer question prompt
    qa_system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Please only use information from the provided documents about 집회 (protests) and provide answers in Korean using respectful language.\ 
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# Streamlit UI
# 로고 경로
path_to_logo = "EchoShieldLogo.png"  # 로고 파일 위치

# Streamlit 헤더에 로고 추가
col1, col2 = st.columns([1, 8])
with col1:
    st.image("EchoShieldLogo.png", width=200)
with col2:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; background-color: #eaf4ff; padding: 1px; border-radius: 8px;">
            <h3 style="color: #003366; font-family: Arial, sans-serif;">집회 시위 참여 시민을 위한 매뉴얼</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


# Streamlit 사이드바 디자인
with st.sidebar:
    st.image("civil-rights.jpg", use_container_width=True)
    st.markdown("### Echo Shield")
    
    # 매뉴얼 소개 Toggle
    with st.expander("💬 매뉴얼 소개"):
        st.write(
            """
            이 서비스는 집회 및 시위 참여 시민을 위해 설계된 매뉴얼입니다. 
            사용자는 이 플랫폼을 통해 다음과 같은 정보를 얻을 수 있습니다:
            
            - 안전, 건강, 응급처치, 편의 시설 등 필요 정보 제공
            - 시민들의 집회 참여에 대한 불안감 해소 및 안전한 참여 촉진
            - 법적 정보 가이드를 통한 시민의 권리 보장
            
            **Echo Shield**는 신뢰할 수 있는 데이터와 실시간 지원을 제공합니다.
            """
        )

    # 질문하기 Toggle
    with st.expander("🔍 질문하기"):
        st.write(
            """
            **질문 창**: 우측 질문 창을 사용하여 즉각적인 답변을 받아보세요.\n
            **E-mail 문의**: 추가적인 질문이나 요청 사항은 아래 이메일로 보내주세요.\n
            📧 **aiffelds@gmail.com**
            """
        )

    # 기타 정보 Toggle
    with st.expander("📜 기타 정보"):
        st.write(
            """
            현재 준비 중입니다. 곧 업데이트될 예정이니 많은 관심 부탁드립니다!
            """
        )

rag_chain = chaining()

# Display messages from session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "집회 중 궁금한게 있으신가요?"}]


if st.button("대화 기록 제거"): # 세션 초기화
    st.session_state["messages"] = [{"role": "assistant", "content": "집회 중 궁금한게 있으신가요?"}]


for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


# Handle user input and generate response
if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
    st.chat_message("human").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt_message)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
