from re import L
from typing import Dict, Any, List , Literal, Callable, Type, Optional
from pydantic import BaseModel, Field , ValidationError
import instructor
import os
from openai import OpenAI
from dotenv import load_dotenv
from tools.search import web_search

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    response_mode: Literal["chat", "websearch"] = Field(
        ...,
        description="Decide whether to respond via chat mode or perform a web search.",
    ) 

def router(query) :
  router_prompt = """
  You are an expert knowing all existing models and AI products out there. 
  - **Chat**: Use this response for general inquiries, FAQs, and straightforward questions that can be answered with your existing knowledge.
  - **Websearch**: Use this response for more complex, real-time, or niche information requests that require specific data or up-to-date information beyond your knowledge.

  Respond with only one word: "chat" if you can answer directly, or "websearch" if the question needs further research.

  """
  router_client = instructor.from_openai(
      OpenAI(
          base_url="http://localhost:11434/v1",
          api_key="ollama",  
      ),
      mode=instructor.Mode.JSON,
  )

  routing = router_client.chat.completions.create(
      model="llama3.1",
      messages=[
          {
              "role": "system",
              "content": router_prompt,
          }
          ,
          {
              "role": "user",
              "content": query,
          }
      ],
      response_model=RouteQuery,
  )

  return routing.response_mode

router(query="HI")

router(query="What do you know about Paligemma model from google ?")

class OllamaChatCompletion:
    """
    Interacts with OpenAI's API for chat completions.
    """
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        """
        Initialize with model, API key, and base URL.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, messages: List[str], tools: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Generates a response from OpenAI's API."""
        params = {'messages': messages, 'model': self.model, 'tools': tools, **kwargs}
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message
    
llm = OllamaChatCompletion(api_key='Empty' ,
                            base_url='http://127.0.0.1:11434/v1' ,
                            model="llama3.1")


# chat message memory
class ChatMessageMemory:
    """Manages conversation context."""

    def __init__(self):
        self.messages = []

    def add_message(self, message: Dict):
        """Add a message to memory."""
        self.messages.append(message)

    def add_messages(self, messages: List[Dict]):
        """Add multiple messages to memory."""
        for message in messages:
            self.add_message(message)

    def add_conversation(self, user_message: Dict, assistant_message: Dict):
        """Add a user-assistant conversation."""
        self.add_messages([user_message, assistant_message])

    def get_messages(self) -> List[Dict]:
        """Retrieve all messages."""
        return self.messages.copy()

    def reset_memory(self):
        """Clear all messages."""
        self.messages = []


class AgentTool:
    """Encapsulates a Python function with Pydantic validation."""
    def __init__(self, func: Callable, args_model: Type[BaseModel]):
        self.func = func
        self.args_model = args_model
        self.name = func.__name__
        self.description = func.__doc__ or self.args_schema.get('description', '')

    def to_openai_function_call_definition(self) -> dict:
        """Converts the tool to OpenAI Function Calling format."""
        schema_dict = self.args_schema
        description = schema_dict.pop("description", "")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": schema_dict
            }
        }

    @property
    def args_schema(self) -> dict:
        """Returns the tool's function argument schema as a dictionary."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def validate_json_args(self, json_string: str) -> bool:
        """Validate JSON string using the Pydantic model."""
        try:
            validated_args = self.args_model.model_validate_json(json_string)
            return isinstance(validated_args, self.args_model)
        except ValidationError:
            return False

    def run(self, *args, **kwargs) -> Any:
        """Execute the function with validated arguments."""
        try:
            # Handle positional arguments by converting them to keyword arguments
            if args:
                sig = signature(self.func)
                arg_names = list(sig.parameters.keys())
                kwargs.update(dict(zip(arg_names, args)))

            # Validate arguments with the provided Pydantic schema
            validated_args = self.args_model(**kwargs)
            return self.func(**validated_args.model_dump())
        except ValidationError as e:
            raise ValueError(f"Argument validation failed for tool '{self.name}': {str(e)}")
        except Exception as e:
            raise ValueError(f"An error occurred during the execution of tool '{self.name}': {str(e)}")

    def __call__(self, *args, **kwargs) -> Any:
        """Allow the AgentTool instance to be called like a regular function."""
        return self.run(*args, **kwargs)

class AgentToolExecutor:
    """Manages tool registration and execution."""

    def __init__(self, tools: Optional[List[AgentTool]] = None):
        self.tools: Dict[str, AgentTool] = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, tool: AgentTool):
        """Registers a tool."""
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self.tools[tool.name] = tool

    def execute(self, tool_name: str, *args, **kwargs) -> Any:
        """Executes a tool by name with given arguments."""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")
        try:
            return tool(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"Error executing tool '{tool_name}': {e}") from e

    def get_tool_names(self) -> List[str]:
        """Returns a list of all registered tool names."""
        return list(self.tools.keys())

    def get_tool_details(self) -> str:
        """Returns details of all registered tools."""
        tools_info = [f"{tool.name}: {tool.description} Args schema: {tool.args_schema['properties']}" for tool in self.tools.values()]
        return '\n'.join(tools_info)
    
logger = logging.getLogger(__name__)

class Agent:
    """Integrates LLM client, tools, memory, and manages tool executions."""

    def __init__(self, llm_client, system_message: Dict[str, str], max_iterations: int = 10, tools: Optional[List[AgentTool]] = None):
        self.llm_client = llm_client
        self.executor = AgentToolExecutor()
        self.memory = ChatMessageMemory()
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.tool_history: List[Dict[str, Any]] = []
        self.function_calls = None

        # Register and convert tools
        if tools:
            for tool in tools:
                self.executor.register_tool(tool)
            self.function_calls = [tool.to_openai_function_call_definition() for tool in tools]

    def run(self, user_message: Dict[str, str]):
        """Generates responses, manages tool calls, and updates memory."""
        self.memory.add_message(user_message)
        derection = router(user_message['content'])

        for _ in range(self.max_iterations):
          if derection == 'websearch':
            chat_history = [self.system_message] + self.memory.get_messages() + self.tool_history
            response = self.llm_client.generate(chat_history, tools=self.function_calls)
            if self.parse_response(response):
                continue
            else:
                self.memory.add_message({"role": "assistant" , "content" : response.content})
                self.tool_history = []
                return response
          else : 
            chat_history = [self.system_message] + self.memory.get_messages()
            response = self.llm_client.generate(chat_history)
            self.memory.add_message({"role": "assistant" , "content" : response.content})
            return response

    def parse_response(self, response) -> bool:
        """Executes tool calls suggested by the LLM and updates tool history."""
        import json

        if response.tool_calls:
            self.tool_history.append(response)
            for tool in response.tool_calls:
                tool_name = tool.function.name
                tool_args = tool.function.arguments
                tool_args_dict = json.loads(tool_args)
                try:
                    logger.info(f"Executing {tool_name} with args: {tool_args}")
                    execution_results = self.executor.execute(tool_name, **tool_args_dict)
                    self.tool_history.append({
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "name": tool_name,
                        "content": str(execution_results)
                    })
                except Exception as e:
                    raise ValueError(f"Execution error in tool '{tool_name}': {e}") from e
            return True
        return False
    
class GetSearchSchema(BaseModel):
    """Fetch and process data from Google search based on a query, store results in ChromaDB vector store, and retrieve results."""
    search_query: str = Field(description="The search query to use for fetching data from Google search")

tools = [
    AgentTool(web_search, 
            GetSearchSchema)
]

# Define the system message
system_message = {"role": "system", 
                  "content": """
                  You are an AI assistant designed to assist users with their questions and inquiries across a wide range of topics. 
                  Your main focus is to answer the user's most recent question directly. You have memory to retain relevant information from previous interactions, which can help provide more personalized responses if needed.
                  Your goal is to deliver accurate, helpful, and concise answers while maintaining a friendly and engaging tone. Feel free to sprinkle in some humor and emojis to make the conversation lively! 
                  Always prioritize clarity, relevance, and user satisfaction in your interactions, utilizing your memory to enhance the user experience when appropriate.
                  """
                  }

agent = Agent(llm_client=llm, 
              system_message=system_message, 
              tools=tools)

# Define a user message
user_message = {"role": "user", "content": "Hi"}

# Generate a response using the agent
response = agent.run(user_message)
print(response.content)

user_message = {"role": "user", "content": "Search the web for information on the Mamba AI architecture and provide details about it"}
response = agent.run(user_message)
print(response.content)

user_message = {"role": "user", "content": "What was your last task ?"}

response = agent.run(user_message)
print(response.content)

user_message = {"role": "user", "content": "Search online and provide information on the latest OpenAI model and project name"}

response = agent.run(user_message)
print(response.content)

agent.memory.get_messages()