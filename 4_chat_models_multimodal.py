import os
import httpx
import base64
import io
from getpass import getpass
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

if __name__ == '__main__':
    # setting
    local_data =  False
    url_b64 = True
    url_stream = False

    # Model
    if 'GOOGLE_API_KEY' not in os.environ:
        os.environ['GOOGLE_API_KEY'] = getpass("Enter the API key: ")
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature = 0.8)

    # Prompt and Payload
    if local_data: 
        # a. Multimodal Inputs: Image base 64 (from  local)
        print("a. Multimodal Inputs: Image base 64 (from  local)")
        image_loc = input("Enter the location of image on local: ")
        image_data = Image.open(image_loc)
        buffer = io.BytesIO()
        image_data.save(buffer, format="jpeg")
        image_bytes = base64.b64encode(buffer.getvalue()).decode('utf-8')
        message1 = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image"
                },
                {
                    "type": "image",
                    "data": image_bytes,
                    "source_type": "base64",
                    "mime_type": "image/jpeg"
                }
            ]
        }
        prompt_template = ChatPromptTemplate([message1])
        payload = {}

    if url_b64:
        # b. Multimodal Inputs: Image base64 (from url)
        print("b. Multimodal Inputs: Image base64 (from url)")
        image_url = "https://www.nps.gov/npgallery/GetAsset/bbbb9e25-091b-4f44-bfda-76bc2f8927e8/proxy/hires?"
        image_data = base64.b64encode(httpx.get(image_url).content).decode('utf-8')
        prompt_template = ChatPromptTemplate(
            [
                {
                    "role": "system",
                    "content": "Describe the image"
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "data": "{image_data}",
                        "source_type": "base64",
                        "mime_type": "image/jpeg"
                    }]
                }
            ]
        )
        payload = {"image_data": image_data}

    if url_stream:
        # c. Multimodal Inputs: Image URL
        print("c. Multimodal Inputs: Image URL")
        image_url = "https://www.nps.gov/npgallery/GetAsset/bbbb9e25-091b-4f44-bfda-76bc2f8927e8/proxy/hires?"

        message3 = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image"
                },
                {
                    "type": "image",
                    "source_type": "url",
                    "url": image_url
                }
            ]
        }
        prompt_template = ChatPromptTemplate([message3])
        payload = {}
        
# generate response 
chain = prompt_template | llm
print(chain.invoke(payload).content)
