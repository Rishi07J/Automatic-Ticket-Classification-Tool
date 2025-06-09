import os
import joblib
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq

from langchain.chains.question_answering import load_qa_chain

def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    PineconeClient(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )
    index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
    return index

def create_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_similar_docs(index, query, k=2):
    return index.similarity_search(query, k=k)

from langchain_groq import ChatGroq

def get_answer(docs, user_input):
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    # Remove OpenAI callback – it's only for OpenAI
    response = chain.run(input_documents=docs, question=user_input)
    return response


def predict(query_result):
    model = joblib.load('modelsvm.pk1')
    return model.predict([query_result])[0]
