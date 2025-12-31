import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


class EcobotRAG:
    def __init__(self, data_path="data/", vector_db_path="vectorstore/faiss_index"):
        self.data_path = data_path
        self.vector_db_path = vector_db_path
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        
    def load_documents(self):
        print(f"Loading documents from {self.data_path}...")
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents):
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")
        return chunks
    
    def create_embeddings(self):
        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Embeddings model loaded")
        return self.embeddings
    
    def create_vector_store(self, chunks):
        print("Creating vector store (this takes 2-3 minutes)...")
        if self.embeddings is None:
            self.create_embeddings()
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        print("Vector store created")
        return self.vectorstore
    
    def save_vector_store(self):
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        self.vectorstore.save_local(self.vector_db_path)
        print(f"Vector store saved to {self.vector_db_path}")
    
    def load_vector_store(self):
        print(f"Loading vector store from {self.vector_db_path}...")
        if self.embeddings is None:
            self.create_embeddings()
        self.vectorstore = FAISS.load_local(
            self.vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded")
        return self.vectorstore
    
    def setup_qa_chain(self, force_ollama=False):
        print("Setting up QA chain...")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if groq_api_key and not force_ollama:
            try:
                print("Attempting to connect to GROQ...")
                from langchain_groq import ChatGroq
                
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=512,
                    groq_api_key=groq_api_key,
                    timeout=10
                )
                
                test_msg = self.llm.invoke("Hi")
                print("GROQ connected successfully")
                
            except Exception as e:
                print(f"GROQ connection failed: {str(e)[:100]}")
                print("Switching to local Ollama backup...")
                self.llm = None
        else:
            print("No GROQ API key found or forced Ollama mode")
            self.llm = None
        
        if self.llm is None:
            try:
                print("Connecting to local Ollama...")
                from langchain_ollama import ChatOllama
                
                self.llm = ChatOllama(
                    model="llama3.2:3b",
                    temperature=0.7
                )
                print("Ollama connected successfully - Backup mode active")
                
            except Exception as e:
                print(f"Ollama also failed: {str(e)}")
                print("Make sure Ollama is running")
                return None
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        print("QA chain ready")
        return True
    
    def query(self, question):
        if self.llm is None:
            raise ValueError("QA chain not setup! Run setup_qa_chain() first.")
        
        docs = self.retriever.invoke(question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are Ecobot, an expert plant care assistant for Snake Plants, Spider Plants, and Aloe Vera in a greenhouse monitoring system.

Based ONLY on the following information, provide a well-structured answer:

{context}

Question: {question}

Instructions:
- Start with a brief 1-2 sentence introduction explaining the cause or situation
- Then provide bullet points (use - for each point) ONLY when listing multiple items, causes, or steps
- For "why" questions: explain the cause first, then list causes as bullets if there are multiple
- For "how to" or "what to do" questions: list solution steps as bullets
- Keep each bullet point concise and actionable
- Maximum 5-6 bullet points
- Be confident and helpful

Answer:"""
        
        response = self.llm.invoke(prompt)
        
        return {'result': response.content}
    
    def chat(self):
        print("\n" + "="*70)
        print("ECOBOT - PLANT CARE ASSISTANT")
        print("="*70)
        print("\nAsk about Snake Plants, Spider Plants, Aloe Vera")
        print("Type 'quit' or 'exit' to stop")
        print("="*70 + "\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Take care of your plants!")
                break
                
            if not question:
                continue
                
            try:
                print("\nEcobot: ", end="", flush=True)
                response = self.query(question)
                print(response['result'])
                print()
            except Exception as e:
                print(f"\nError: {str(e)}\n")


def setup_from_scratch():
    print("\n" + "="*70)
    print("ECOBOT - FIRST TIME SETUP")
    print("="*70 + "\n")
    
    bot = EcobotRAG()
    
    documents = bot.load_documents()
    chunks = bot.split_documents(documents)
    
    bot.create_embeddings()
    bot.create_vector_store(chunks)
    bot.save_vector_store()
    
    print("\nSetup complete!\n")
    return bot


def load_existing_bot():
    print("\n" + "="*70)
    print("LOADING ECOBOT")
    print("="*70 + "\n")
    
    bot = EcobotRAG()
    
    try:
        bot.load_vector_store()
        bot.setup_qa_chain()
        print("\nEcobot ready to chat!\n")
        return bot
    except Exception as e:
        print(f"\nError loading bot: {e}")
        print("Try deleting the 'vectorstore' folder and running setup again.\n")
        return None


if __name__ == "__main__":
    vector_exists = os.path.exists("vectorstore/faiss_index")
    
    if not vector_exists:
        print("\nNo existing vector database found.")
        print("Creating new vector database...")
        print("This will take 2-3 minutes...\n")
        
        response = input("Proceed with setup? (y/n): ").strip().lower()
        if response == 'y':
            bot = setup_from_scratch()
            bot.setup_qa_chain()
            
            chat = input("\nStart chatting now? (y/n): ").strip().lower()
            if chat == 'y':
                bot.chat()
        else:
            print("\nSetup cancelled.")
    else:
        bot = load_existing_bot()
        if bot:
            bot.chat()