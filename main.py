import sys
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatAnthropic(model="claude-3-5-haiku")

# response = llm.invoke("What are the capabilities of Claude? What are the best use cases?")
# print(response)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are Sreedeep, a customer support executive that will help in resolving users query
            and complaints. Answer the user query and use neccessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("What can I help you with?")
try:
    raw_response = agent_executor.invoke({"query": query})
    # print(raw_response)
except Exception as e:
    print(f"API call failed: {e}")
    sys.exit(1)


try:
    structured_response = parser.parse(raw_response.get("output")[0].get("text"))
except Exception as e:
    print(f"Error parsing response {e} Raw Response - {raw_response}")

print(structured_response)