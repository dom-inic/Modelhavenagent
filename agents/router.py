from typing import Dict, Any, List , Literal
from pydantic import BaseModel, Field , ValidationError
import instructor
import os
from openai import OpenAI
from dotenv import load_dotenv

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
          api_key=os.getenv("OPENAI_API_KEY"),  # required, but unused
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