from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# MessagesPlaceholder, SystemMessage, HumanMessage (with functions)
system_message = SystemMessage("You are a helpful assistant")
placeholder_message = MessagesPlaceholder("inp")
prompt_template = ChatPromptTemplate.from_messages([system_message, placeholder_message])
print(prompt_template.invoke({"inp": [HumanMessage("Tell me a joke about Snakes")]}))

# MessagesPlaceholder (with tuples)
system_message = ("system", "You are a helpful assistant")
placeholder_message = ("placeholder", "{inp}")
prompt_template = ChatPromptTemplate.from_messages([system_message, placeholder_message])
print(prompt_template.invoke({"inp": [HumanMessage("Tell me a joke about Snakes")]}))
