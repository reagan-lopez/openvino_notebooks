# Create llama-2-chat-7b Chatbot using OpenVINO
In Windows command prompt:
- Run `.\setup.bat`
- Run `python llm-chatbot-prepare` for preparing the models. Do this just once.
- Run `python llm-chatbot-inference` for inferencing.
  If you did not run the prepare step, at lease make sure to copy `llm_config.py` from `../../utils`