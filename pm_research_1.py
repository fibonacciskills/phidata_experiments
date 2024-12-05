from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
import inspect
import types

# Define individual agents
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[GoogleSearch()],
    instructions=[
        "Given a topic by the user, respond with 4 latest news items about that topic.",
        "Search for 10 news items and select the top 4 unique items.",
        "Search in English and in French.",
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

tech_market_agent = Agent(
    name="Technology and Market Opportunity Expert",
    role="Analyze technology trends, market dynamics, and identify value creation opportunities",
    model=OpenAIChat(id="gpt-4o"),
    tools=[GoogleSearch()],
    instructions=[
        "Analyze emerging technology trends and market dynamics",
        "Identify potential market opportunities",
        "Evaluate competitive landscapes",
        "Always include data-driven insights",
        "Always include sources"
    ],
    show_tool_calls=True,
    markdown=True,
)

value_capture_agent = Agent(
    name="Value Capture Strategist",
    role="Develop strategies for IP protection, market positioning, and competitive advantage",
    model=OpenAIChat(id="gpt-4o"),
    tools=[GoogleSearch()],
    instructions=[
        "Focus on IP protection strategies",
        "Develop market positioning recommendations",
        "Identify competitive advantages",
        "Provide actionable strategic recommendations",
        "Always include sources"
    ],
    show_tool_calls=True,
    markdown=True,
)

org_design_agent = Agent(
    name="Organizational Design Architect",
    role="Design optimal organizational structures and collaboration networks",
    model=OpenAIChat(id="gpt-4o"),
    tools=[GoogleSearch()],
    instructions=[
        "Design team structures and collaboration frameworks",
        "Optimize for innovation and value delivery",
        "Consider organizational culture and dynamics",
        "Provide practical implementation steps",
        "Always include sources"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Combine agents into a team
agent_team = Agent(
    team=[
        web_agent, 
        finance_agent, 
        tech_market_agent, 
        value_capture_agent, 
        org_design_agent
    ],
    instructions=[
        "Always include sources",
        "Use tables to display data",
        "Provide comprehensive product management strategies",
        "Integrate market, technology, and organizational perspectives",
        "Focus on both value creation and value capture",
        "Include actionable recommendations"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Function signature inspection utility
def inspect_function(func):
    if callable(func):
        if isinstance(func, types.BuiltinFunctionType):
            print("Cannot inspect signature of built-in type.")
        else:
            try:
                argspec = inspect.signature(func)
                print(f"Signature: {argspec}")
            except ValueError:
                print("Signature could not be retrieved.")
    else:
        raise ValueError("Provided object is not callable.")

# Function to ask the user for input and run the agent team
def start_business_analysis():
    # Ask the user for their input
    business_type = input("What kind of company do you want to create? ")
    
    # Use the input to tailor the analysis
    prompt = f"Analyze the opportunities, challenges, and strategies for creating a company focused on '{business_type}'. Provide comprehensive recommendations on market positioning, financial planning, technology trends, organizational design, and value capture."
    
    # Debugging: Print the prompt
    print("Prompt for analysis:", prompt)
    
    # Execute the agent team with the customized prompt
    response = agent_team.print_response(prompt, stream=True)
    
    # Debugging: Check the response content
    if not response:
        print("Received an empty response from the agent team.")
    else:
        print("Response received:", response)

# Start the process
start_business_analysis()

def test_web_agent():
    test_prompt = "Search for the top 5 latest news items about AI startups."
    response = web_agent.print_response(test_prompt, stream=True)
    print("Web Agent Response:", response)

# Call the test function
test_web_agent()