from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate 

# 1. String Prompt Template
prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
print(prompt_template.invoke({"topic": "Cats"}))

# 2.a. Chat Prompt Template (with Tuples)
system_message = ("system", "You are a helpful assistant")
human_message = ("human", "Tell me a joke about {topic}")
prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])
print(prompt_template.invoke({"topic": "Dogs"}))

# 2.b. Chat Prompt Template (with MessagePromptTemplate)
system_template = SystemMessagePromptTemplate.from_template("You are a helpful assistant")
user_template = HumanMessagePromptTemplate.from_template("Tell me a joke about {topic}")
prompt_template = ChatPromptTemplate([system_template, user_template])
print(prompt_template.invoke({"topic": "Birds"}))