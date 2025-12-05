# RAG Chatbot Assignment

A Python **Retrieval-Augmented Generation (RAG) Chatbot** that answers questions based on transactional data.  
Supports both **CLI** and **Streamlit Web UI** using the OpenAI API.


##  Setup Instructions

### 1️. Clone the Repository

```bash
git clone https://github.com/Cyberider1008/rag-chatbot-python.git
cd rag-chatbot-python
```

### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3️. Create .env File
Inside the project root, create a file named **.env**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
## Run CLI Chatbot
```bash
python main.py
```
Type any question.<br>
Type **exit** or **quit** to stop the chatbot.

## Run Streamlit Web App
```bash
streamlit run app.py
```
Then open in browser:
```bash
http://localhost:8501
```
## Example Questions to Ask
- What is Amit’s total spending?
- Show me Riya’s purchase history.
