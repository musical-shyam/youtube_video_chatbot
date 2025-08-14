# Import necessary libraries for the YouTube bot
import gradio as gr
from yt_io import *
from config import *
from rag import *

# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Step 1: Set up IBM Watson credentials
        model_id, credentials, client, project_id = setup_credentials()

        # Step 2: Initialize WatsonX LLM for summarization
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."

def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up IBM Watson credentials
        model_id, credentials, client, project_id = setup_credentials()

        # Step 3: Initialize WatsonX LLM for Q&A
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model(credentials, project_id)
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."

with gr.Blocks(title="YouTube RAG Assistant") as interface:
    gr.Markdown("# YouTube Video Summarizer and Q&A Chatbot")
    gr.Markdown("This chatbot can summarize YouTube videos and answer questions about their content.\n")
    gr.Markdown("### Enter the YouTube video URL to get started:")
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")

    # Outputs for summary
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    # Button for summarizing video
    summarize_btn = gr.Button("Summarize Video")
    

    
    # Input field for user question
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    # Outputs for question and answer
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)
    # Buttons for selecting functionalities after fetching transcript
    question_btn = gr.Button("Ask a Question")
    
    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="127.0.0.1", server_port=7860, share=True, debug=True)


# # Sample YouTube URL
# url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# video_id = get_video_id(url)
# # print(video_id)  # Output: dQw4w9WgXcQ

# # Fetching the transcript
# transcript = get_transcript(url)
# # print(transcript)

# # Formating the transcript
# formatted_transcript = process(transcript)
# # print(formatted_transcript)

# # Chunking the transcript
# chunks = chunk_transcript(formatted_transcript)
# # print(chunks)

# # Creating the Q&A prompt template 
# qa_prompt_template = create_qa_prompt_template()

# # Example of how to use the prompt template with context and a question
# context = "This video explains the fundamentals of quantum physics."
# question = "What are the key principles discussed in the video?"

# # Generating the prompt
# generated_prompt = qa_prompt_template.format(context=context, question=question)

# # Output the generated prompt
# #print(generated_prompt)