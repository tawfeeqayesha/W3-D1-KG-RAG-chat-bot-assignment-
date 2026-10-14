import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

# --- Page Setup ---
st.set_page_config(
    page_title="Policy KG-RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# --- Secure Credentials Retrieval ---
# Pulls from Streamlit Secrets or environment variables if configured
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

st.title("🤖 Policy Knowledge Graph Assistant")
st.write(
    "Ask questions about your policy document using Neo4j Knowledge Graphs and Guardrails."
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Document & Database Config")
    pdf_path = st.text_input("Policy PDF File", value="KG_RAG_project/KG Rag doc.pdf")
    
    st.markdown("---")
    st.subheader("Neo4j Aura Credentials")
    neo4j_url = st.text_input("Neo4j URI", value="neo4j+s://your-instance-id.databases.neo4j.io")
    neo4j_user = st.text_input("Neo4j Username", value="neo4j")
    neo4j_pass = st.text_input("Neo4j Password", type="password", value="")
    neo4j_db = st.text_input("Neo4j Database Name", value="neo4j")
    
    if st.button("Process & Ingest into Neo4j", type="secondary"):
        if not openai_api_key:
            st.error("OpenAI API Key is missing. Add it to Streamlit secrets or your environment.")
        elif not os.path.exists(pdf_path):
            st.error(f"File '{pdf_path}' not found. Check path.")
        else:
            with st.spinner("Loading PDF, extracting graph entities, and storing in Neo4j..."):
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    st.info(f"Loaded {len(docs)} pages. Extracting graph nodes...")

                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
                    transformer = LLMGraphTransformer(llm=llm)
                    graph_documents = transformer.convert_to_graph_documents(docs)

                    graph = Neo4jGraph(
                        url=neo4j_url.strip(),
                        username=neo4j_user.strip(),
                        password=neo4j_pass.strip(),
                        database=neo4j_db.strip()
                    )
                    graph.add_graph_documents(
                        graph_documents,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    
                    st.success(f"Successfully indexed {len(docs)} pages into Neo4j Knowledge Graph!")
                except Exception as e:
                    st.error("Ingestion failed due to the following error:")
                    st.exception(e)

    st.markdown("---")
    st.markdown("### Architecture Pipeline")
    st.write("- **Graph Store:** Neo4j Knowledge Graph")
    st.write("- **Guardrails:** Input validation & Output compliance check")

# --- Guardrail Rules ---
def input_guardrail(query: str) -> bool:
    prohibited_terms = ["hack", "exploit", "bypass", "malware", "system prompt", "jailbreak"]
    query_lower = query.lower()
    for term in prohibited_terms:
        if term in query_lower:
            return False
    return True

def output_guardrail(answer: str) -> str:
    if not answer or len(answer.strip()) < 10:
        return "I could not verify this information within the official policy document."
    return answer

# --- Query Interface ---
question = st.text_area(
    "Enter your question",
    placeholder="Example: What are the main objectives outlined in the policy?",
    height=100,
)

if st.button("Find policy information", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not openai_api_key:
        st.error("OpenAI API Key is required to run queries.")
    elif not input_guardrail(question):
        st.error("Input Guardrail Triggered: The query contains restricted or non-compliant text.")
    else:
        with st.spinner("Querying Neo4j Knowledge Graph and generating answer..."):
            try:
                graph = Neo4jGraph(
                    url=neo4j_url.strip(),
                    username=neo4j_user.strip(),
                    password=neo4j_pass.strip(),
                    database=neo4j_db.strip()
                )
                
                cypher_query = "MATCH (d:Document) RETURN d.text AS text LIMIT 5"
                results = graph.query(cypher_query)
                
                if not results:
                    st.warning("No documents found in Neo4j. Please click 'Process & Ingest into Neo4j' first.")
                else:
                    context = "\n\n".join([r["text"] for r in results if r.get("text")])

                    prompt = ChatPromptTemplate.from_template(
                        """You are an official AI assistant for policy analysis.
                        Answer the user's question using ONLY the retrieved context below.
                        If the answer cannot be determined from the context, clearly state that the information is unavailable.

                        Context:
                        {context}

                        Question: {question}
                        """
                    )

                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
                    chain = (
                        {"context": lambda x: context, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    raw_answer = chain.invoke(question)
                    final_answer = output_guardrail(raw_answer)

                    st.subheader("Answer")
                    st.markdown(final_answer)

                    st.subheader("Retrieved Graph Context")
                    for r in results:
                        st.info(r["text"][:300] + "...")

            except Exception as exc:
                st.error("Unable to query Neo4j. Verify your database is active and credentials are correct.")
                st.exception