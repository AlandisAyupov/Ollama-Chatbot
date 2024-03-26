# chainlit run app_chainlit_ollama.py

# Langchain: Langchain is an open-source framework designed to make it easier for developers to build applications that use large language models (LLMs).  
# Think of it as a set of tools that helps developers  work with LLMs to create things like chatbots, document analysis applications, 
# and more.

# chainlit: Chainlit is an open-source Python framework specifically designed to help developers build conversational AI applications quickly and 
# easily. It simplifies the process by providing tools to handle many of the common tasks involved in creating chatbots and similar applications.

# Imports the Ollama class from the langchain_community.llms module. 
# Ollama is a class that allows you to interact with large language models (LLMs) through Langchain.
from langchain_community.llms import Ollama

# ChatPromptTemplate class from the langchain.prompts module. 
# This class is used to define the format of the prompts that will be sent to the LLM.
from langchain.prompts import ChatPromptTemplate

# This line imports the StrOutputParser class from the langchain.schema module. 
# This class is likely used to parse the output that is received from the LLM into a string format.
from langchain.schema import StrOutputParser

# Defining how the chatbot application itself works within the Langchain framework.
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

#defines function that starts on chat session
@cl.on_chat_start
async def on_chat_start():
  # Sending an image with the local file path
  # Creates an list containing 1 element of the cl.Image class.
  elements = [ cl.Image(name="image1", display="inline", path="images/mistral_logo.jpg")  ]
  # AI message. Contains the image.
  await cl.Message(content="Hello there, I am Mistral. How can I help you ?", elements=elements).send()
  # Configures Ollama model.
  model = Ollama(model="mistral:latest")
  # Configures prompt.
  prompt = ChatPromptTemplate.from_messages( [
    ( "system",
      "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
    ),
    ("human", "{question}"),
  ] )
  # "|"" represents pipes. Prompt -> model -> StrOutputParser()
  runnable = prompt | model | StrOutputParser()
  # Sets chat session data for the specific user in question.
  cl.user_session.set("runnable", runnable)

# --------------------------------------------------------------
#Handles incoming messages
@cl.on_message
async def on_message(message: cl.Message):
  runnable = cl.user_session.get("runnable")  # Get runnable input.

  # Initialize empty message
  msg = cl.Message(content="")

  # Iterates over chunks in runnable stream,
  async for chunk in runnable.astream(
    {"question": message.content},
    config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
  ):
    await msg.stream_token(chunk)

  #Sends message
  await msg.send()