# Youtube Video Summarizer and Q/A Chatbot
YouTube Video Summarizer and Q&A Chatbot can summarize YouTube videos and answer questions about their content using AI. Follow the following steps to start the 

# Step 1: Install Necessary Libraries
pip install youtube-transcript-api==1.2.1
pip install faiss-cpu==1.8.0
pip install langchain==0.2.6 
pip install langchain-community==0.2.6 
pip install ibm-watsonx-ai==1.0.10 
pip install langchain_ibm==0.1.8 
pip install gradio==4.44.1 

# Step 2: Store WatsonX.AI API keys
$env:IBM_API_KEY="<ADD YOUR IBM API KEY>"
$env:IBM_PROJECT_ID="<ADD YOUR IBM PROJECT ID>"
visit https://medium.com/@harangpeter/setting-up-ibm-watsonx-ai-for-api-based-text-inference-435ef6d1a6a3 to get started setting up your api key and project id

# Step 3: Run the Application in the Background
python ytbot.py

# Step 4: Launch the application in web browser
paste the following link in a web browser: http://127.0.0.1:7860