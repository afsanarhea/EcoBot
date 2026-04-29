import sys
sys.path.insert(0, ".")
from ecobot_rag import EcobotRAG

bot = None

def call_api(prompt, options, context):
    global bot
    
    if bot is None:
        bot = EcobotRAG()
        bot.load_vector_store()
        bot.setup_qa_chain()
    
    try:
        response = bot.query(prompt)
        return {
            "output": response["result"]
        }
    except Exception as e:
        return {
            "output": f"Error: {str(e)}"
        }
