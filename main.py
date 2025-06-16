from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig

import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

agent = Agent(
    name = "Customer Service Agent",
    instructions = "You are a helpful customer service agent. Answer the user's questions politely and accurately.",
)

@cl.on_chat_start
async def hendle_chat_start():
    cl.user_session.set( "history", [] )
    await cl.Message(content = "Hello from Ayesha Nasir! I am here to assist you with your questions." ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({
        "role": "user",
        "content": message.content
    })

    result = await Runner.run(
        agent,
        input = history,
        run_config = config,
    )

    history.append({
        "role": "assistant",
        "content": result.final_output
    })
    cl.user_session.set("history", history)

    await cl.Message(content = result.final_output).send()