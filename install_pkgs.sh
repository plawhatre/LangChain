# Install the Ollama Python package
source env_langchain/bin/activate
# Install the required Python packages
python3 -m pip install -r requirements.txt
python3 -m pip install langchain-community pypdf
python -m pip install faiss-cpu
# vector db
pip install --upgrade setuptools
pip install protobuf==6.30.0
python -m pip install langchain-chroma>=0.1.2
python -m pip install rank_bm25