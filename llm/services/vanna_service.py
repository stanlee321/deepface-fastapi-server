# vanna_service.py
import sqlite3
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from configs import settings

# Define your custom Vanna class
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # Initialize ChromaDB in a specific path (e.g., ./chroma_db)
        # You can customize the path as needed
        chroma_config = {'path': './chroma_db'}
        if config and 'chromadb_path' in config:
            chroma_config['path'] = config['chromadb_path']
        
        ChromaDB_VectorStore.__init__(self, config=chroma_config)
        OpenAI_Chat.__init__(self, config=config)

_vanna_instance = None

def get_processed_descriptions_ddl(db_path: str = 'llm_descriptions.db') -> str | None:
    """Fetches the DDL for the processed_descriptions table."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='processed_descriptions';")
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            print("Table 'processed_descriptions' not found.")
            return None
    except Exception as e:
        print(f"Error fetching DDL: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_processed_descriptions_ddl_2(db_path: str = 'llm_descriptions.db') -> str | None:
    """Fetches the DDL for the processed_descriptions table."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='raw_descriptions';")
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            print("Table 'raw_descriptions' not found.")
            return None
    except Exception as e:
        print(f"Error fetching DDL: {e}")
        return None
    finally:
        if conn:
            conn.close()

def initialize_vanna():
    """Initializes and trains the Vanna instance."""
    global _vanna_instance
    if _vanna_instance is not None:
        return _vanna_instance

    config = {
        'api_key': settings.OPENAI_API_KEY,
        'model': settings.OPENAI_MODEL, # Or your preferred OpenAI model
        'chromadb_path': './my_vanna_db' # Optional: customize ChromaDB path
    }
    
    print("Initializing Vanna instance...")
    vn = MyVanna(config=config)
    
    print("Connecting Vanna to SQLite database (llm_descriptions.db)...")
    vn.connect_to_sqlite('llm_descriptions.db')
    
    print("Fetching DDL for 'processed_descriptions' table...")
    ddl_1 = get_processed_descriptions_ddl()
    ddl_2 = get_processed_descriptions_ddl_2()
    if ddl_1 and ddl_2:
        print(f"Training Vanna with DDL: {ddl_1}")
        vn.train(ddl=ddl_1)
        vn.train(ddl=ddl_2)
    else:
        print("Could not train Vanna with DDL as it was not found.")

    # Optional: Add some documentation about the table or columns
    vn.train(documentation="The processed_descriptions table contains descriptions that have been processed by an LLM, along with their status and associated codes.")
    vn.train(documentation="The 'processed_description' column contains the text of the description after LLM processing.")
    vn.train(documentation="The 'code' column is a unique identifier associated with the description.")
    vn.train(documentation="The 'status' column indicates the current state of the description (e.g., 'processed', 'pending').")
    
    # Optional: Add some example question-SQL pairs if you have common queries
    # vn.train(question="How many processed descriptions have status 'pending'?", 
    #          sql="SELECT COUNT(*) FROM processed_descriptions WHERE status = 'pending';")

    _vanna_instance = vn
    print("Vanna instance initialized and trained.")
    return _vanna_instance

def get_vanna() -> MyVanna:
    """Dependency function to get the Vanna instance."""
    global _vanna_instance
    if _vanna_instance is None:
        # This case should ideally be handled by lifespan, but as a fallback:
        print("Vanna instance not ready, attempting to initialize...")
        initialize_vanna()
        if _vanna_instance is None: # Check again after attempt
             raise RuntimeError("Vanna instance could not be initialized.")
    return _vanna_instance