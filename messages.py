from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# MessagesPlaceholder, SystemMessage, HumanMessage
system_message = SystemMessage("You are a helpful assistant")
placeholder = MessagesPlaceholder("inp")
prompt_template = ChatPromptTemplate.from_messages([system_message, placeholder])
print(prompt_template.invoke({"inp": [HumanMessage("Tell me a joke about Snakes")]}))