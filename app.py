import os
import zipfile
from pathlib import Path
import boto3
import json



from langchain_core.documents import Document

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_openai import ChatOpenAI


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

ZIP_FILE = BASE_DIR / "LIC DOCS.zip"

PDF_FOLDER = BASE_DIR / "LICc"
# AWS S3 Configuration

S3_BUCKET = "idp-rag-documents-akshitha-20261014"
AWS_REGION = "ap-south-1"

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION
)


# ============================================================
# 2. LOAD .ENV
# ============================================================

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY"
)


# ============================================================
# 3. STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Insurance Assistant",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 10px;
        font-size: 17px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🛡️ AI Insurance Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        RAG-Powered Insurance Policy & Claims Assistant
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. API KEY CHECK
# ============================================================

if not OPENAI_API_KEY:

    st.error(
        "⚠️ OPENAI_API_KEY is not configured."
    )

    st.info(
        f"""
        The .env file should be located here:

        {ENV_FILE}

        Inside .env add:

        OPENAI_API_KEY=your_api_key_here
        """
    )

    st.stop()


# ============================================================
# 7. EXTRACT ZIP FILE
# ============================================================

def prepare_documents():

    # --------------------------------------------------------
    # Check if LICc folder already has PDFs
    # --------------------------------------------------------

    if PDF_FOLDER.exists():

        pdf_files = list(
            PDF_FOLDER.rglob("*.pdf")
        )

        if pdf_files:

            return True

    # --------------------------------------------------------
    # Extract ZIP
    # --------------------------------------------------------

    if ZIP_FILE.exists():

        PDF_FOLDER.mkdir(
            exist_ok=True
        )

        try:

            with zipfile.ZipFile(
                ZIP_FILE,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    PDF_FOLDER
                )

            return True

        except zipfile.BadZipFile:

            st.error(
                "❌ LIC DOCS.zip is not a valid ZIP file."
            )

            return False

    return False





# =====================================================
# LOAD EXTRACTED INSURANCE TEXT FROM S3
# =====================================================

def load_documents_from_s3():
    documents = []

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix="extracted-text/"
    ):
        for item in page.get("Contents", []):
            key = item["Key"]

            if not key.lower().endswith(".json"):
                continue

            response = s3.get_object(
                Bucket=S3_BUCKET,
                Key=key
            )

            data = json.loads(
                response["Body"].read().decode("utf-8")
            )

            source_key = data.get("source_key", key)

            for page_data in data.get("pages", []):
                text = page_data.get("text", "").strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_key,
                            "page": int(page_data["page"]) - 1
                        }
                    )
                )

    return documents


# ============================================================
# 9. LOAD INSURANCE DOCUMENTS FROM S3
# ============================================================

@st.cache_resource(
    show_spinner="📄 Loading insurance documents from S3..."
)
def load_documents():

    # Load extracted JSON files from S3
    documents = load_documents_from_s3()

    if not documents:
        return [], []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return documents, chunks


documents, chunks = load_documents()


if not documents:

    st.error(
        "No extracted insurance text was found in S3."
    )

    st.info(
        "Check the extracted-text/ folder in your S3 bucket."
    )

    st.stop()
    
# ============================================================
# 10. EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner="🧠 Loading embedding model..."
)
def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )

    return embeddings


embeddings = create_embeddings()


# ============================================================
# 11. CREATE CHROMA VECTOR STORE
# ============================================================
#
# IMPORTANT:
# No pickle file.
# No Insurance_RAG folder.
# No persistent database.
#
# Chroma is created in memory.
#
# ============================================================

@st.cache_resource(
    show_spinner="🗄️ Creating vector database..."
)
def create_vector_store():

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


vector_store = create_vector_store()


# ============================================================
# 12. CREATE HYBRID RETRIEVER
# ============================================================

@st.cache_resource(
    show_spinner="🔎 Preparing hybrid retriever..."
)
def create_hybrid_retriever():

    # --------------------------------------------------------
    # BM25 RETRIEVER
    # --------------------------------------------------------

    bm25_retriever = BM25Retriever.from_documents(
        chunks
    )

    bm25_retriever.k = 5

    # --------------------------------------------------------
    # MMR RETRIEVER
    # --------------------------------------------------------

    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    # --------------------------------------------------------
    # ENSEMBLE RETRIEVER
    # --------------------------------------------------------

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            mmr_retriever
        ],
        weights=[
            0.4,
            0.6
        ]
    )

    return ensemble_retriever


retriever = create_hybrid_retriever()


# ============================================================
# 13. CREATE OPENAI LLM
# ============================================================

@st.cache_resource
def create_llm():

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    return llm


llm = create_llm()


# ============================================================
# 14. HYDE
# ============================================================

def generate_hyde(question):

    hyde_prompt = f"""
Write a short factual paragraph that is likely to appear
in an insurance policy document, claims guideline, or
customer support knowledge base answering the given question.

Do not invent policy details.
Do not answer conversationally.
Use clear insurance terminology.

Question:

{question}

Hypothetical insurance document paragraph:
"""

    response = llm.invoke(
        hyde_prompt
    )

    return response.content


# ============================================================
# 15. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question):

    # --------------------------------------------------------
    # Generate HyDE document
    # --------------------------------------------------------

    hypothetical_document = generate_hyde(
        question
    )

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    retrieved_docs = retriever.invoke(
        hypothetical_document
    )

    return retrieved_docs


# ============================================================
# 16. GENERATE FINAL ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_docs
):

    # --------------------------------------------------------
    # Prepare context
    # --------------------------------------------------------

    context_parts = []

    for doc in retrieved_docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            None
        )

        if page is not None:

            source_name = (
                f"{Path(str(source)).name}"
                f" - Page {page + 1}"
            )

        else:

            source_name = Path(
                str(source)
            ).name

        context_parts.append(
            f"[Source: {source_name}]\n"
            f"{doc.page_content}"
        )

    context_text = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Insurance RAG Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI Insurance Assistant.

Answer ONLY from the retrieved insurance documents.

STRICT RULES

1. Use ONLY the provided context.

2. Do NOT use your own knowledge about insurance.

3. Do NOT infer, assume, or guess any information.

4. Do NOT add information that is not explicitly
   mentioned in the context.

5. If the context is insufficient, answer EXACTLY:

The provided insurance documents do not contain enough
information to answer this question.

6. If the answer exists in the context, provide a
   clear, concise and factual response.

7. When applicable, mention policy terms, coverage,
   exclusions, eligibility, deductibles, premiums,
   claim procedures, or required documents exactly
   as stated in the context.

8. Do not provide legal or financial advice beyond
   the retrieved information.

Context:

{context_text}

Question:

{question}

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🛡️ Insurance Assistant"
    )

    st.write(
        "Ask questions about insurance "
        "policy documents."
    )

    st.divider()

    st.subheader(
        "📊 Knowledge Base"
    )

    st.metric(
        "Documents",
        len(documents)
    )

    st.metric(
        "Text Chunks",
        len(chunks)
    )

    st.divider()

    st.subheader(
        "🔧 RAG Pipeline"
    )

    st.write(
        "👤 User Question"
    )

    st.write("↓")

    st.write(
        "🧠 HyDE"
    )

    st.write("↓")

    st.write(
        "🔎 BM25 + MMR"
    )

    st.write("↓")

    st.write(
        "⚖️ Ensemble Retrieval"
    )

    st.write("↓")

    st.write(
        "📚 Retrieved Context"
    )

    st.write("↓")

    st.write(
        "🤖 GPT-4.1-mini"
    )

    st.divider()

    st.caption(
        "RAG answers are generated only "
        "from retrieved insurance documents."
    )


# ============================================================
# 18. EXAMPLE QUESTIONS
# ============================================================

st.subheader(
    "💡 Example Questions"
)

example_questions = [

    "What is the minimum entry age for LIC's Single Premium Endowment Plan?",

    "What are the premium payment options in LIC's Amritbaal?",

    "When can a policy loan be availed?",

    "What is the free-look period for this policy?",

    "Can the policyholder surrender the policy before maturity?"

]


columns = st.columns(5)


for i, question in enumerate(
    example_questions
):

    with columns[i]:

        if st.button(
            f"Example {i + 1}",
            use_container_width=True
        ):

            st.session_state[
                "question"
            ] = question


# ============================================================
# 19. USER QUESTION
# ============================================================

st.subheader(
    "🔎 Ask Your Insurance Question"
)

question = st.text_area(
    "Enter your question",
    value=st.session_state.get(
        "question",
        ""
    ),
    placeholder=(
        "Example: What is the free-look "
        "period for this policy?"
    ),
    height=110
)


# ============================================================
# 20. ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Get Insurance Answer",
    type="primary",
    use_container_width=True
)


# ============================================================
# 21. RUN RAG
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "⚠️ Please enter an insurance question."
        )

    else:

        # ----------------------------------------------------
        # STEP 1 — HYDE
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Understanding your question..."
        ):

            retrieved_docs = retrieve_documents(
                question
            )

        # ----------------------------------------------------
        # STEP 2 — CHECK RESULTS
        # ----------------------------------------------------

        if not retrieved_docs:

            st.warning(
                "No relevant information was found "
                "in the insurance documents."
            )

        else:

            # ------------------------------------------------
            # STEP 3 — GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "🤖 Generating insurance answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_docs
                )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Insurance Assistant"
            )

            st.markdown(
                f"""
                <div class="answer-box">
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📚 Retrieved Sources"
            )

            for i, doc in enumerate(
                retrieved_docs,
                1
            ):

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page"
                )

                source_name = Path(
                    str(source)
                ).name

                if page is not None:

                    source_label = (
                        f"{source_name}"
                        f" — Page {page + 1}"
                    )

                else:

                    source_label = source_name

                with st.expander(
                    f"📄 Source {i}: {source_label}"
                ):

                    st.write(
                        doc.page_content
                    )


# ============================================================
# 22. FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ AI Insurance Assistant | "
    "RAG | HyDE | BM25 | MMR | ChromaDB"
)