import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from mcp_use import MCPAgent, MCPClient
import os

async def run_memory_chat():
    """Run a chat using MCPAgents's built in conversation memory."""
    #Load environment variable for API keys

    load_dotenv()
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    #config file path

    config_file="browser_mcp.json"

    print("Initializing chat")

    # Create MCP client and agent with memory enabled

    #Client
    client=MCPClient.from_config_file(config_file)

    #LLM
    llm=ChatGroq(model="qwen/qwen3-32b")

    # Create agent with memory_enabled =true

    agent=MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True


    )

    print("\n==Interactive MCP Chat==")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("====================================\n")

    try:
        # MAin chat loop

        while True:
            #GEt user input
            user_input=input("\nYou: ")

            #check for exit command

            if user_input.lower() in ["exit","quit"]:
                print("Ending conversation.")
                break

            #check for clear history command:

            if user_input.lower() =="clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue
            
            #Get response from agent

            print("\nAssistant: ", end="",flush=True)

            try:
                #Run the agent with the user input(Memory handling is automatic)
                response=await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"\nError : {e}")

    finally:
        #clean up

        if client and client.sessions:
            await client.close_all_sessions()

if __name__=="__main__":
    asyncio.run(run_memory_chat())
