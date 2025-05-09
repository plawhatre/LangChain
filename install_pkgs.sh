# Install the Ollama Python package
source env_langchain/bin/activate
# Install the required Python packages
python3 -m pip install -r requirements.txt
python3 -m pip install langchain-community pypdf
python3 -m pip install langchain-chroma
python -m pip install faiss-cpu