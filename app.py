import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# --- Page and Sidebar Setup ---
st.set_page_config(page_title="LangChain Chat with Search", page_icon="🔎")
st.title("🔎 LangChain - Chat with search")
st.info(
    "In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app. "
    "Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent)."
)

with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your Groq API Key:", type="password")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# --- Main App Logic ---
if not api_key:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# --- Initialize Tools and Agent (only once) ---
try:
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    # Arxiv and Wikipedia Tools
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # DuckDuckGo Search Tool
    search_tool = DuckDuckGoSearchRun(name="Search")

    tools = [search_tool, arxiv_tool, wiki_tool]

    # Initialize the agent with error handling
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
except Exception as e:
    st.error(f"Failed to initialize the agent. Please check your API key and settings. Error: {e}")
    st.stop()


# --- Handle User Input ---
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Wrap the agent call in a try-except block to handle runtime errors
        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
        except Exception as e:
            # Display a friendly error message in the UI if the agent fails
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({'role': 'assistant', "content": error_message})
