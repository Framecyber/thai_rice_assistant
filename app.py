import gradio as gr
from langchain_groq import ChatGroq
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import PromptTemplate
from neo4j import GraphDatabase
import networkx as nx
import pinecone
import os

# RAG Setup
text_path = r"C:\Users\USER\Downloads\RAG_langchain\text_chunks.txt"
loader = TextLoader(text_path, encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY', '6396a319-9bc0-49b2-97ba-400e96eff377'),
    environment='gcp-starter'
)

index_name = "langchain-demo"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=768)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
else:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

rag_llm = ChatGroq(
    model="Llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    groq_api_key='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'
)

rag_prompt = PromptTemplate(
    template="""
    You are a Thai rice assistant that gives concise and direct answers. 
    Do not explain the process, 
    just provide the answer,
    provide the answer only in Thai."

    Context: {context}
    Question: {question}
    Answer: 
    """,
    input_variables=["context", "question"]
)

rag_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)

graphrag_llm = ChatGroq(
    model="Llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    groq_api_key='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'
)

uri = "neo4j+s://46084f1a.databases.neo4j.io"
user = "neo4j"
password = "FwnX0ige_QYJk8eEYSXSF0l081mWWGIS7TFg6t8rLZc"
driver = GraphDatabase.driver(uri, auth=(user, password))

def fetch_nodes(tx):
    query = "MATCH (n) RETURN id(n) AS id, labels(n) AS labels"
    result = tx.run(query)
    return result.data()

def fetch_relationships(tx):
    query = "MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relation"
    result = tx.run(query)
    return result.data()

def populate_networkx_graph():
    G = nx.Graph()
    with driver.session() as session:
        nodes = session.read_transaction(fetch_nodes)
        relationships = session.read_transaction(fetch_relationships)
        for node in nodes:
            G.add_node(node['id'], labels=node['labels'])
        for relationship in relationships:
            G.add_edge(
                relationship['source'],
                relationship['target'],
                relation=relationship['relation']
            )
    return G

networkx_graph = populate_networkx_graph()
graph = NetworkxEntityGraph()
graph._graph = networkx_graph

graphrag_chain = GraphQAChain.from_llm(
    llm=graphrag_llm,
    graph=graph,
    verbose=True
)

def get_rag_response(question):
    response = rag_chain.invoke(question)
    return response

def get_graphrag_response(question):
    system_prompt = "You are a Thai rice assistant that gives concise and direct answers. Do not explain the process, just provide the answer, provide the answer only in Thai."
    formatted_question = f"System Prompt: {system_prompt}\n\nQuestion: {question}"
    response = graphrag_chain.run(formatted_question)
    return response

def compare_models(question):
    rag_response = get_rag_response(question)
    graphrag_response = get_graphrag_response(question)
    return rag_response, graphrag_response

def store_feedback(feedback, question, rag_response, graphrag_response):
    print("Storing feedback...")
    print(f"Question: {question}")
    print(f"RAG Response: {rag_response}")
    print(f"GraphRAG Response: {graphrag_response}")
    print(f"User Feedback: {feedback}")
    
    with open("feedback.txt", "a", encoding='utf-8') as f:
        f.write(f"Question: {question}\n")
        f.write(f"RAG Response: {rag_response}\n")
        f.write(f"GraphRAG Response: {graphrag_response}\n")
        f.write(f"User Feedback: {feedback}\n\n")

def handle_feedback(feedback, question, rag_response, graphrag_response):
    store_feedback(feedback, question, rag_response, graphrag_response)
    return "Feedback stored successfully!"

with gr.Blocks() as demo:
    gr.Markdown("## Thai Rice Assistant A/B Testing")

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Ask a question about Thai rice:")
            submit_btn = gr.Button("Get Answers")

        with gr.Column():
            rag_output = gr.Textbox(label="Model A", interactive=False)
            graphrag_output = gr.Textbox(label="Model B", interactive=False)

    with gr.Row():
        with gr.Column():
            choice = gr.Radio(["A is better", "B is better", "Tie", "Both Bad"], label="Which response is better?")
            send_feedback_btn = gr.Button("Send Feedback")

    def on_submit(question):
        rag_response, graphrag_response = compare_models(question)
        return rag_response, graphrag_response

    def on_feedback(feedback):
        question = question_input.value
        rag_response = rag_output.value
        graphrag_response = graphrag_output.value
        return handle_feedback(feedback, question, rag_response, graphrag_response)

    submit_btn.click(on_submit, inputs=[question_input], outputs=[rag_output, graphrag_output])
    send_feedback_btn.click(on_feedback, inputs=[choice], outputs=[])

demo.launch(share=True)
