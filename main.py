from litellm import completion
import os
import instructor
import litellm
from openai import OpenAIError
# from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import  OpenAIEmbeddings
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_fixed

os.environ["ANTHROPIC_API_KEY"] = "bad-key"

llm_client = instructor.from_litellm(completion)

# file_path = (
#     "Book3.pdf"
# )
# loader = PyPDFLoader(file_path)

# docs = loader.load()

with open("data.json", "r") as file:
    data = json.load(file)

# Combine fields like title and text
texts = [f"{doc['title']}.\n{doc['text']}" for doc in data]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.create_documents(texts)

vectorstore = Chroma.from_documents(documents = texts, embedding = OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# def track_cost_callback(
#     kwargs,                 # kwargs to completion
#     completion_response,    # response from completion
#     start_time, end_time    # start/end time
# ):
#     try:
#       response_cost = kwargs.get("response_cost", 0)
#       print("streaming response_cost", response_cost)
#     except:
#         pass

# litellm.success_callback = [track_cost_callback] 

class CircularActionables(BaseModel):
    actionable_title: str = Field(..., description="Requirement of calculator or specific compliance")
    actionable_manner_of_compliance: List[str] = Field(..., description="Steps required to calculate or reach the final answer.")
    confidence_score: float = Field(..., description="Confidence score of the answer being generated.")
    final_answer: Optional[str] = Field(None, description="The final answer after retries or strategic guessing.")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_response(prompt : str) -> CircularActionables :
    try:
        response = llm_client.chat.completions.create(
        model="claude-3-5-sonnet-20241022", 
        messages=[{"role": "user", "content": prompt}],
        response_model=CircularActionables,
        temperature = 0.0)

        if response.confidence_score > 0.75:
            return response
        else:
            raise ValueError("Confidence score too low, retrying....")
    except Exception as e:
        print(f"Error during response generation: {e}")
        raise

try:
    query = """Hey, how's it going?"""
    results = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    final_response = generate_response(prompt)
except Exception as e:
    ValueError("Unable to generate even after retries")
    print(e)