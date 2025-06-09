import argparse
import json
import logging
import os
import warnings

from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM

from salesgpt.agents import SalesGPT

load_dotenv()  # loads .env file

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress logging
logging.getLogger().setLevel(logging.CRITICAL)

# LangSmith settings section, set TRACING_V2 to "true" to enable it
# or leave it as it is, if you don't need tracing (more info in README)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = ""  # insert you project name here

def main():
    """Run the SalesGPT agent."""
    
    # Initialize the language model
    llm = ChatLiteLLM(temperature=0.2, model_name="gpt-3.5-turbo")
    
    # Initialize the sales agent
    sales_agent = SalesGPT(
        stage_id=1,
        salesperson_name="Ted Lasso",
        salesperson_role="Business Development Representative", 
        company_name="Sleep Haven",
        company_business="Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible.",
        company_values="Our mission at Sleep Haven is to help people achieve a better night's sleep by providing exceptional mattresses at an affordable price.",
        conversation_purpose="find out whether they are looking to achieve better sleep via buying a premier mattress.",
        conversation_type="call",
        use_tools=False,
        product_catalog="sample_product_catalog.csv",
        llm=llm,
        verbose=True,
    )
    
    # Seed the agent
    sales_agent.seed_agent()
    
    print("SalesGPT Agent Initialized. Starting conversation...")
    print("=" * 50)
    
    # Start the conversation
    sales_agent.step()
    
    # Interactive conversation loop
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
            
        sales_agent.human_step(user_input)
        sales_agent.determine_conversation_stage()
        sales_agent.step()


if __name__ == "__main__":
    main()
