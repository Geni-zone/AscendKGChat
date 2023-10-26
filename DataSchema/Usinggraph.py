import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
from io import BytesIO
from typing import List
import openai
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl

openai.api_key = ""
openai_api_key = ""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.

Example of your response should be:

The answer is foo
SOURCES: xyz


Begin!
----------------
{summaries}"""


@cl.on_chat_start
async def on_chat_start():
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf", "text/plain"],
            max_size_mb=100,
            timeout=180,
            max_files=5
        ).send()

    chunks = []
    for file in files:
        _, ext = os.path.splitext(file.name)
        print(f"File type: {ext}")

        msg = cl.Message(
                content=f"Processing `{file.name}`...", disable_human_feedback=True
            )
        await msg.send()

        
        if ext == ".txt":
            tex = file.content.decode("utf-8")
            chunks.extend(text_splitter.split_text(text=tex))
        elif ext == ".pdf":
            print("entered pdf elif")
            pdf_file = PyPDF2.PdfReader(BytesIO(file.content))

            # Get the total number of pages in the PDF file.
            num_pages = len(pdf_file.pages)

            # Create a list to store the decoded text from each page.
            decoded_text = []

            # Decode each page of the PDF file and add the decoded text to the list.
            for i in range(num_pages):
                page = pdf_file.pages[i]
                decoded_text.append(page.extract_text())

            for text in decoded_text:
                chunks.extend(text_splitter.split_text(text))


    metadatas = [{"source": f"{i}-pl"} for i in range(len(chunks))]

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_model="text-embedding-ada-002")

    documents_embedded = embeddings.embed_documents(chunks)


    G = nx.Graph()
    if os.path.exists('graph.json'):
        with open("graph.json", "r") as f:
            data = json.load(f)
            G = data

    for idx, (text, embedding, source) in enumerate(zip(chunks, documents_embedded, metadatas)):
        G.add_node(idx, text=text, embedding=embedding, source=source)
    for idx1, data1 in G.nodes(data=True):
        for idx2, data2 in G.nodes(data=True):
            if idx1 != idx2:
                similarity = cosine_similarity([data1['embedding']], [data2['embedding']])[0][0]
                if similarity > 0.75:  
                    G.add_edge(idx1, idx2, weight=similarity)
    data = nx.node_link_data(G)
    with open("graph.json", "w") as f:
        json.dump(data, f)

    
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("Graph", G)


@cl.on_message
async def main(message: cl.Message):
    graph = cl.user_session.get("Graph")  # type: ConversationalRetrievalChain

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_model="text-embedding-ada-002")
    userembed = embeddings.embed_documents(message.content)
    resentry, resneighbors = query_graph(userembed, graph)
    neighbor_texts = [neighbor['text'] for neighbor in resneighbors]

    # Joining neighbor texts into a single string with newline separator
    neighbor_texts_str = '\n'.join(neighbor_texts)

    # Updating the prompt to include all neighbor texts
    prompt = f"{system_template}\n\nContext: {resentry['text']}\n{neighbor_texts_str}\n\nUser: {message.content}\n"
    print(prompt)
    messages = [
        {"role": "system", "content": "You will use the context in the user input to generate factual output about new information"}
        ]
    finalprompt = {"role": "user", "content": prompt}
    messages.append(finalprompt)
    completion = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = messages,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    result = completion["choices"][0]["message"]
    answer = result['content']
    source_documents =[]    # type: List[Document]
    source_documents.append(resentry['source'])
    for neighbor in resneighbors:
        source_documents.append(neighbor['source'])
    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc['source'], name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()



def query_graph(query_embedding, G):
    # Ensure query_embedding is a numpy array
    query_embedding = np.array(query_embedding).flatten()

    # Determine the smallest dimension among all embeddings
    min_dimension = min(
        [len(data['embedding']) for idx, data in G.nodes(data=True)],
        default=len(query_embedding)
    )

    # Slice each embedding and the query_embedding to have the same size
    similarities = [
        (
            idx,
            cosine_similarity(
                [np.array(data['embedding']).flatten()[:min_dimension]],
                [query_embedding[:min_dimension]]
            )[0][0]
        )
        for idx, data in G.nodes(data=True)
    ]
    entry_point_node_idx, _ = max(similarities, key=lambda x: x[1])

    # Gather nodes that are one relation away
    neighbors = list(G.neighbors(entry_point_node_idx))
    
    # Return the entry point node and its neighbors
    return G.nodes[entry_point_node_idx], [G.nodes[neighbor_idx] for neighbor_idx in neighbors]

