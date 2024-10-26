"""
@fileoverview
This module implements the Retrieval-Augmented Generation (RAG) functionality for the Business Sector Chatbot.
It leverages FAISS for vector similarity search, SentenceTransformer for embeddings, and FlashRank for re-ranking.
The module processes user queries, determines the necessity of database access, retrieves relevant documents,
and streams the generated response back to the client.

@version 1.0
"""

import re
import json
import logging
import warnings
import requests

# FAISS imports for vector similarity search
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Ranker import for re-ranking search results
from flashrank import Ranker, RerankRequest

# Suppress any warnings to keep the logs clean
warnings.filterwarnings("ignore")

# Initialize logging with INFO level and a specific format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize embedding function using SentenceTransformer
logging.debug("Initializing embedding function...")
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
logging.info("Embedding function initialized.")

# Load FAISS vector stores for full HR dataset and QA HR dataset
logging.debug("Loading FAISS vector stores...")
faiss_Full_HR = FAISS.load_local("Prototype/Backend/Database/HR/Vector/Full_HR", embedding_function, allow_dangerous_deserialization=True)
faiss_QA_HR = FAISS.load_local("Prototype/Backend/Database/HR/Vector/QA_HR", embedding_function, allow_dangerous_deserialization=True)
logging.info("FAISS vector stores loaded.")

# Initialize the ranker for re-ranking search results
logging.debug("Initializing the ranker...")
ranker = Ranker(model_name="rank-T5-flan", cache_dir="/Temp")
logging.info("Ranker initialized.")


def generate_stream(payload):
    """
    Generates a streaming response based on the provided payload using Retrieval-Augmented Generation (RAG).

    Args:
        payload (dict): A dictionary containing the following keys:
            - 'model' (str): Identifier for the model to use.
            - 'messages' (list): List of message dictionaries containing 'role' and 'content'.
            - 'options' (dict): Optional parameters such as 'temperature', 'num_predict', and 'num_ctx'.
            - 'stream' (bool): Indicates whether to stream the response.
            - 'keep_alive' (int): Determines if the connection should be kept alive.

    Yields:
        str: Streaming chunks of the generated response or error messages in JSON format.

    Raises:
        ValueError: If the 'messages' format in the payload is invalid.
    """
    logging.info("Starting generate_stream...")
    # Extract model, messages, options from payload
    model = payload.get('model', 'default-model')
    messages = payload.get('messages', [])
    options = payload.get('options', {})
    temperature = options.get('temperature', 0.8)
    max_tokens = options.get('num_predict', int(4096))
    context_length = options.get('num_ctx', int(8192))
    stream = payload.get('stream', True)
    logging.debug(f"Model: {model}, Temperature: {temperature}, Max Tokens: {max_tokens}, Stream: {stream}")

    # Validate messages format
    if not messages or not isinstance(messages, list):
        logging.error("Invalid messages format in payload.")
        raise ValueError('Invalid messages format')

    # Extract the user's latest message
    user_message = messages[-1].get('content', '')
    logging.debug(f"User message: {user_message}")
    # Build chat history excluding the last message
    chat_history = messages[:-1]

    try:

        ''' Function/Tool calls working in Refractor
        # Define the available functions
                available_functions = [
                    {
                        "name": "get_employee_data",
                        "description": "Fetch specific personal data fields of an employee from the database.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "fields": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ALLOWED_FIELDS
                                    },
                                    "description": "List of employee data fields to retrieve. Allowed fields are: employee_id, name, department, job_title, salary, leaves_taken_this_month."
                                }
                            },
                            "required": ["fields"]
                        }
                    },
                    {
                        "name": "get_hr_policy",
                        "description": "Fetch HR policy information by querying the vector database.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_query": {
                                    "type": "string",
                                    "description": "The query related to HR policy that is to be similarity searched in HR dataset."
                                }
                            },
                            "required": ["user_query"]
                        }
                    }
                ]

        '''

        # Step 1: Get user query (user_message already obtained)
        logging.info("Step 1: User query obtained.")

        # Step 2: Perform a self-query to determine if database access is required
        logging.info("Step 2: Performing self-query to determine if database is required.")
        # Prepare the self-query prompt
        self_query_prompt = f"""
Given the following question:

"{user_message}"

Context:
 You are Enerzal, a friendly and intelligent chatbot developed by Tech Enerzal. Your primary role is to assist employees of Tech Enerzal by providing helpful, polite, and accurate information. You should always maintain a friendly and approachable tone while ensuring your responses are clear and informative. Your purpose is to assist with the following:

1. **HR-Related Queries:** Help employees with questions regarding company policies, leave management, employee benefits, payroll, and other HR-related topics. Be empathetic and supportive, especially for sensitive topics like leave or benefits.

2. **IT Support:** Provide guidance on common IT issues employees may encounter, such as troubleshooting technical problems, resetting passwords, or navigating company software. Be patient and provide step-by-step instructions for resolving technical issues.

3. **Company Events & Updates:** Keep employees informed about upcoming company events, milestones, and internal updates. Share details about events in a friendly, enthusiastic tone to keep the company culture vibrant and engaging.

4. **Uploaded Document Summarization and Querying:** Enerzal also helps employees by summarizing documents (PDF, DOCX, TXT) and answering queries based on the content of uploaded documents. For document summaries, be concise and informative, extracting the key points while maintaining clarity. When answering queries, provide clear and accurate answers based on the document content, making sure to offer further assistance if needed.

Determine whether the assistant needs to access an external database for only  HR , IT , Company events to provide an accurate answer. For Uploaded Document's and casual talks Default to NO 

Answer with 'Yes' if the database is required, or 'No' if the database is not required.

Answer in the following format:

"Database required: Yes" or "Database required: No"
"""
        logging.debug(f"Self-query prompt: {self_query_prompt}")

        # Call the gemma2:2b model API for the self-query
        # Prepare payload for the self-query
        self_query_model_api_url = 'http://localhost:11434/api/chat'  # Replace with your actual gemma2:2b API endpoint
        self_query_model_payload = {
            'model': 'gemma2:2b',  # Specify the model
            'messages': [
                {'role': 'system', 'content': 'You are an assistant that determines whether a database is required to answer a question.'},
                {'role': 'user', 'content': self_query_prompt}
            ],
            'options': {
                'temperature': 0.0,
                "num_predict": int(15),
            },
            'stream': False,  # Self-query does not need streaming
            'keep_alive': 0
        }
        logging.debug(f"Self-query model payload: {self_query_model_payload}")

        # Make the API call to the self-query model
        logging.info(f"Making self-query API call to {self_query_model_api_url}")
        self_query_response = requests.post(self_query_model_api_url, json=self_query_model_payload)
        self_query_response.raise_for_status()  # Raise an exception for HTTP errors
        self_query_data = self_query_response.json()
        logging.debug(f"Self-query response data: {self_query_data}")

        # Parse the response to determine if database access is required
        if 'message' in self_query_data:
            message = self_query_data['message']
            if isinstance(message, dict):
                assistant_reply = message.get('content', '').strip()
            else:
                logging.error(f"Invalid message format in response: {message}")
                raise ValueError(f'Invalid message format in response: {message}')
        elif 'messages' in self_query_data:
            messages = self_query_data['messages']
            if messages and isinstance(messages, list):
                assistant_reply = messages[-1].get('content', '').strip()
            else:
                logging.error(f"Invalid messages format in response: {messages}")
                raise ValueError(f'Invalid messages format in response: {messages}')
        else:
            logging.error(f"No message or messages key found in response: {self_query_data}")
            raise ValueError(f'No message or messages key found in response: {self_query_data}')
        logging.debug(f"Assistant reply from self-query: {assistant_reply}")

        # Extract the database requirement from the assistant's reply using regex
        match = re.search(r'Database required:\s*(Yes|No)', assistant_reply, re.IGNORECASE)
        if match:
            database_required = match.group(1).strip().lower() == 'yes'
            logging.info(f"Database required: {database_required}")
        else:
            logging.warning(f"Could not parse database requirement from assistant reply: {assistant_reply}")
            database_required = False  # Default to False if parsing fails

        # Conditional logic based on whether the database is required
        # For now, keep the category and type determination commented out
        # If database is required, determine the type and category (to be implemented)
        # Example:
        # if database_required:
        #     # Extend the self-query prompt to ask for type and category
        #     self_query_prompt += """
        # Also, specify which database type is required (currently only HR) and select the Section category from one of these for HR:
        # (Recruitment Policy, Appointments and Promotions, Leave and Attendance, Performance Review, General Conduct, Ethics & Disciplinary Action, Medical Reimbursement and Facilities, Grievance Policy, Retirement and Resignation, Training & Development, Insurance Policies, Allowances & Benefits, Housing Policies)
        # Provide the answer in the following format:
        # "Database required: Yes; Type: HR; Category: Recruitment Policy"
        # """
        if database_required:
            logging.info("Database is required. Proceeding to search vector DB and generate response with context.")
             # Proceed to search vector DB and generate response with context

            # Step 3: Query the full HR dataset (faiss_Full_HR)
            logging.info("Step 3: Querying the full HR dataset.")
            k_full = 10  # Number of documents to retrieve
            full_hr_candidates = faiss_Full_HR.similarity_search(user_message, k=k_full)
            logging.debug(f"Retrieved {len(full_hr_candidates)} documents from full HR dataset.")

            # Step 4: Query the QA of the top 2 selected Sections from Full HR
            logging.info("Step 4: Querying the QA of the top 2 selected sections from Full HR.")
            top_sections = full_hr_candidates[:2]
            section_names = [doc.metadata.get('section_name') for doc in top_sections]
            logging.debug(f"Top section names: {section_names}")

            # Retrieve related FAQs from faiss_QA_HR based on the top sections
            qa_candidates = []
            for section_name in section_names:
                logging.debug(f"Querying FAQs for section: {section_name}")
                # Use the 'filter' parameter in FAISS similarity_search to narrow down the search
                k_qa = 10
                qa_results = faiss_QA_HR.similarity_search(
                    user_message,
                    k=k_qa,
                    filter={'section_name': section_name}
                )
                qa_candidates.extend(qa_results)
                logging.debug(f"Retrieved {len(qa_results)} FAQs for section {section_name}")
                logging.debug(f"Retrieved {qa_results} FAQs for section {section_name}")

            # Step 5: Re-rank the QA passages to select the most relevant FAQs
            logging.info("Step 5: Re-ranking the QA passages.")
            # Re-rank the QA passages
            qa_passages = [{
                'id': doc.metadata.get('ids', ''),
                'text': doc.page_content,
                'meta': doc.metadata
            } for doc in qa_candidates]
            logging.debug(f"Total QA passages for re-ranking: {len(qa_passages)}")

            rerank_request = RerankRequest(query=user_message, passages=qa_passages)
            reranked_qa_results = ranker.rerank(rerank_request)
            logging.debug("Reranked QA results obtained.")

            # Select top FAQs after re-ranking
            top_faqs = reranked_qa_results[:3]
            logging.info(f"Selected top {len(top_faqs)} FAQs.")

            # Step 6: Combine the retrieved sections and FAQs to prepare the context
            logging.info("Step 6: Preparing context and modifying messages.")
            # Prepare context from the top sections
            context_sections = '\n\n'.join([doc.page_content for doc in top_sections])
            # Prepare context from the top FAQs
            context_faqs = '\n\n'.join([faq['text'] for faq in top_faqs])

            # Currently, only sections are included in the context
            # context = f"Sections:\n{context_sections}\n\nFAQs:\n{context_faqs}"

            context = f"Sections:\n{context_sections}\n"

            logging.debug("Context prepared.")

            # Insert system message with context before the last user message
            messages.insert(-1, {
                'role': 'system',
                'content': f'Using the provided context from the database for Tech Enerzal to answer the user query.\nContext="{context}"'
            })
            logging.debug("Inserted system message with context into messages.")

        else:
            logging.info("Database is not required. Proceeding without context.")
            # No need to modify messages if database access is not required

        # Step 7: Call the model via API to generate the final response
        logging.info("Step 7: Calling the model via API.")
        # Prepare payload for the model API
        model_api_url = 'http://localhost:11434/api/chat'  # Replace with your actual model API endpoint
        model_payload = {
            'model': model,
            'messages': messages,  # Use the modified messages list
            'options': {
                'temperature': temperature,
                "num_predict": max_tokens,
                "num_ctx": context_length,
            },
            # 'stream': stream,  # Uncomment if streaming is required
            'keep_alive': 0
        }
        logging.debug(f"Model API payload prepared with messages.")

        # Function to stream the response from the model API
        def stream_model_response():
            """
            Streams the response from the model API by making a POST request and yielding response chunks.

            Yields:
                str: JSON-formatted response chunks or error messages.
            """
            logging.debug(f"Making model API call to {model_api_url}")
            with requests.post(model_api_url, json=model_payload, stream=True) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                logging.info("Model API call successful. Streaming response...")
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        # Assuming the API returns JSON lines with 'message' containing 'role' and 'content'
                        try:
                            data = json.loads(decoded_line)
                            logging.debug(f"Received JSON data: {data}")
                    
                            # Accessing nested 'message' object
                            message = data.get('message', {})
                            role = message.get('role')
                            content = message.get('content', '')

                            # Only yield user and assistant messages
                            if role in ['assistant', 'user']:
                                yield content + '\n'
                                logging.debug(f"Yielded content chunk: {content}")
                            else:
                                logging.debug(f"Ignored message with role: {role}")
                        
                        except json.JSONDecodeError:
                            # If the line is not valid JSON, log it and yield the raw line
                            logging.warning(f"Received non-JSON line: {decoded_line}")
                            yield decoded_line + '\n'

                        # Log the raw line received for additional visibility
                        logging.debug(f"Raw line received: {decoded_line}")

        # Yield the response chunks to the caller
        for chunk in stream_model_response():
            yield chunk

    except Exception as e:
        logging.exception("Error in generate_stream")
        yield json.dumps({"error": f"Error in generating response: {str(e)}"})

