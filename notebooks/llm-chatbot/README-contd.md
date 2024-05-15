# Create LLM Chatbot using OpenVINO
In Windows command prompt:
- Follow the installation instructions at `https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows`
- Navigate to this directory: `cd notebooks\llm-chatbot`
- Install the required packages: `python -m pip install -r requirements.txt`
- Prepare the models: `python llm-chatbot-prepare.py --model_idx 2 --precisions INT4,INT8`
- Run the chat app: `python llm-chatbot-inference.py --model_idx 2 --precision INT4 --device GPU`