import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool
import re
import ast
from typing import List, Tuple,Any, Optional

# Streamlit app
st.set_page_config(page_title="Database Assistant", layout="wide")

st.markdown("<h1 style='text-align: center;'> Database Assistant</h1>", unsafe_allow_html=True)
# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Visualize", "Table","Table_Description","Related_tables"])

# Sidebar for database connection
st.sidebar.header("Database Connection")
db_url = st.sidebar.text_input("Database URL")
db_name = st.sidebar.text_input("Database Name")
port=st.sidebar.text_input("Port Number")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'table_names' not in st.session_state:
    st.session_state.table_names = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'toolkit' not in st.session_state:
    st.session_state.toolkit = None
if 'sql_search_tool' not in st.session_state:
    st.session_state.sql_search_tool = None

# Connect button
if st.sidebar.button("Connect"):
    try:
        db_uri = f"mysql+pymysql://{username}:{password}@{db_url}:{port}/{db_name}"
        engine = create_engine(db_uri)
        db = SQLDatabase(engine)
        st.session_state.connected = True
        st.session_state.db = db
        st.session_state.engine = engine
        st.session_state.table_names = db.get_usable_table_names()
        st.sidebar.success("Connected successfully!")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {str(e)}")
        
config = {
            "azure": {
                "api_key": "YOUR-API-KEY-HERE",
                "azure_endpoint": "https://sharkvision-swedencentral.openai.azure.com/",
                "api_version": "2024-05-01-preview",
                "deployment_name": "gpt-4o-mini"
                }
            }
            
llm = AzureChatOpenAI(
    api_key=config["azure"]["api_key"],
    azure_endpoint=config["azure"]["azure_endpoint"],
    api_version=config["azure"]["api_version"],
    deployment_name=config["azure"]["deployment_name"],
    temperature=0.0,
    model_name="gpt-4o-mini"
)

MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: 'Explanation:'. Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.
- If the User Query is not related to the database and if you are not able to find the User query
in the database, just return 'I don\'t know' as the answer, the requested query is not present in the database.

"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.


"""

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_deployment="text-embedding-3-small",
    api_key=config["azure"]["api_key"],
    azure_endpoint=config["azure"]["azure_endpoint"],
    api_version="2023-05-15"
)

def extract_table_desc(table_names, top=10):
    table_descriptions = {}

    for table_name in table_names:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=st.session_state.engine)
            # Replace non-JSON-compliant values with None
            df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
            
            column_info = {}
            sparse_columns = []
            normal_columns = []

            for col in df.columns:
                unique_values = df[col].unique()[:top+1].tolist()
                column_info[col] = unique_values
                if len(unique_values) > top:
                    sparse_columns.append(col)
                else:
                    normal_columns.append(col)
                   
            table_descriptions[table_name] = {
                "columns": column_info,
                "sparse_columns": sparse_columns,
                "normal_columns": normal_columns,
            }
        except Exception as e:
            table_descriptions[table_name] = {"error": str(e)}
    
    return table_descriptions

def create_vector_store(db: SQLDatabase) -> FAISS:
    # Create a vector store for proper nouns in the database
    proper_nouns = []
    table_names=selected_tables
    proper_nouns += table_names
    for table in table_names:
        query = f"SHOW COLUMNS FROM {table}"
        results = db.run(query)
        res = [el for sub in ast.literal_eval(results) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        for val in res:
            if val == 'name':
                query = f"SELECT {val} FROM {table}"
                results = db.run(query)
                res = [el for sub in ast.literal_eval(results) for el in sub if el]
                res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]  
                proper_nouns+=res
    
    vector_store=FAISS.from_texts(proper_nouns, embeddings)
    return vector_store


def sql_search_tool(query: str = None) -> str:
    """
    Search for the most similar term or phrase in the database.

    Args:
        query (str, optional): Keyword or phrase to search for. Defaults to None.

    Returns:
        str: Most similar term found or an error message.
    """
    global retriever

    if not query:
        return "No input provided"
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"score_threshold": 0.85,"k": 1})
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return f"No similar terms found for '{query}'"
    
    return relevant_docs[0].page_content

def find_related_tables_and_columns(table_names):
    result = {}
    for table_name in table_names:
        table_info = {}
        
        # Query to get related tables
        related_tables_query = f"""
            SELECT 
                REFERENCED_TABLE_NAME 
            FROM 
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
            WHERE 
                TABLE_NAME = '{table_name}' 
                AND REFERENCED_TABLE_NAME IS NOT NULL;
        """
        related_tables_df = pd.read_sql_query(related_tables_query, con=st.session_state.engine)
        table_info['related_tables'] = related_tables_df['REFERENCED_TABLE_NAME'].unique().tolist()
        
        # Query to get column names
        columns_query = f"""
            SELECT 
                COLUMN_NAME 
            FROM 
                INFORMATION_SCHEMA.COLUMNS 
            WHERE 
                TABLE_NAME = '{table_name}';
        """
        columns_df = pd.read_sql_query(columns_query, con=st.session_state.engine)
        table_info['columns'] = columns_df['COLUMN_NAME'].tolist()
        
        result[table_name] = table_info

    return result

def find_similar_terms(query: str) -> List[Tuple[str, str]]:
    """
    Find similar terms for each word in the query.

    Args:
        query (str): The input query string.

    Returns:
        List[Tuple[str, str]]: List of tuples containing original words and their similar terms.
    """
    words = re.findall(r'\w+|"[^"]*"', query)
    similar_terms = []

    for word in words:
        word = word.strip('"')
        result = sql_search_tool(word)
        if result != f"No similar terms found for '{word}'":
            similar_terms.append((word, result))

    return similar_terms

def restructure_query(query: str, similar_terms, llm: AzureChatOpenAI) -> str:
    prompt = f"""
    Original query: {query}
    Similar terms found: {similar_terms}
    
    Please reformulate the query by replacing the original terms with the most similar terms found in the database. 
    Follow these rules:
    1. Preserve the original meaning and intent of the query.
    2. Only replace terms if there's a clear, similar match in the database.
    3. Keep the structure and wording of the original query as much as possible.
    4. Do not add new concepts or change the type of information being requested.
    5. Only change the Nouns like names of people, names of the columns and tables. Do not try to replace everything.

    Reformulated query:
    """
    response = llm.invoke(prompt).content
    return response

def query_pipeline(user_query: str, sql_search_tool: sql_search_tool, llm: AzureChatOpenAI) -> str:
    similar_terms = find_similar_terms(user_query)
    restructured_query = restructure_query(user_query, similar_terms, llm)
    return restructured_query


def initialize_db_and_tools():
    if st.session_state.connected and selected_tables:
        st.session_state.db = SQLDatabase(engine=st.session_state.engine, include_tables=selected_tables)
        st.session_state.vector_store = create_vector_store(st.session_state.db)
        st.session_state.toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=llm)
        st.session_state.sql_search_tool = sql_search_tool
        st.sidebar.success(f"Database updated with {len(selected_tables)} selected tables.")
    else:
        st.sidebar.warning("No tables selected. Please select at least one table.")

def extract_python_code(response: str) -> Optional[str]:
    # Look for code blocks
    code_blocks = re.findall(r'```(?:python)?(.*?)```', response, re.DOTALL)
    
    if code_blocks:
        # If code blocks are found, use the first one
        return code_blocks[0].strip()
    else:
        # If no code blocks, try to extract Python-like code
        lines = response.split('\n')
        code_lines = []
        for line in lines:
            if re.match(r'^[\s]*[a-zA-Z_][\w\s]*=.*', line) or \
               re.match(r'^[\s]*import\s+\w+', line) or \
               re.match(r'^[\s]*from\s+\w+\s+import', line):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
    
    return None

def create_dataframe(code: str) -> Optional[pd.DataFrame]:
    try:
        local_vars = {}
        exec(code, globals(), local_vars)
        df = local_vars.get('df')
        if isinstance(df, pd.DataFrame):
            return df
        else:
            raise ValueError("The code did not create a DataFrame named 'df'")
    except Exception as e:
        st.error(f"Error creating DataFrame: {str(e)}")
        return None
    
def process_query_and_display_result(agent: Any, llm: Any, table_query: str) -> None:

    """
    Process a query, generate a DataFrame, and display it using Streamlit.
    
    Args:
    agent (Any): The agent object used to invoke the query.
    llm (Any): The language model object used to generate DataFrame code.
    query (str): The query to process.
    """

    try:
        table_query_response = agent.invoke(table_query)['output']
        
        table_code_query = f"""Generate a suitable pandas DataFrame code for the query result: {table_query_response} 
                               1. Only include the Python code to create the DataFrame.
                               2. Do not include any print statements or display commands.
                               3. Ensure the DataFrame is named 'df'.
                               4. Do not hallucinate the values for the code
                               5. If there are no values or anything that can be change into data 
                               4. Wrap the code in triple backticks."""
        table_response = llm.invoke(table_code_query).content
        print("LLM Response:", table_response)
        
        code = extract_python_code(table_response)
        if code is None:
            st.error("No valid Python code found in the response.")
            return
        
        print("Extracted Code:", code)
        
        df = create_dataframe(code)
        if df is not None:
            st.dataframe(df)
        else:
            st.warning("Unable to create DataFrame.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Table selection
if st.session_state.connected:
    st.sidebar.header("Table Selection")
    
    # Create a dictionary to store the state of each checkbox
    if 'table_checkboxes' not in st.session_state:
        st.session_state.table_checkboxes = {table: False for table in st.session_state.table_names}
    
    # Checkbox for "Select All"
    select_all = st.sidebar.checkbox("Select All Tables")
    
    # Update all checkboxes if "Select All" is toggled
    if select_all:
        for table in st.session_state.table_checkboxes:
            st.session_state.table_checkboxes[table] = True
    elif 'select_all_checked' not in st.session_state or not st.session_state.select_all_checked:
        for table in st.session_state.table_checkboxes:
            st.session_state.table_checkboxes[table] = st.sidebar.checkbox(
                table, 
                value=st.session_state.table_checkboxes[table]
            )
    
    selected_tables = [table for table, is_selected in st.session_state.table_checkboxes.items() if is_selected]

    # Initialize the database and tools only if the selected tables change
    if 'selected_tables' not in st.session_state or selected_tables != st.session_state.selected_tables:
        if selected_tables:
            initialize_db_and_tools()
            st.session_state.selected_tables = selected_tables
        else:
            st.sidebar.warning("No tables selected. Please select at least one table.")
else:
    st.warning("Please connect to a database first.")


with tab1:
    if st.session_state.connected and st.session_state.toolkit and st.session_state.sql_search_tool:
        # User input
        chat_query = st.text_input("Enter your question:", key=10)
        
        # if chat_query:
        #     restructured_query = query_pipeline(chat_query, st.session_state.sql_search_tool, llm)
        #     st.subheader("Restructured Query")
        #     edited_query = st.text_area("Edit the restructured query if needed:", value=restructured_query)

        agent = create_sql_agent(
                        prefix=MSSQL_AGENT_PREFIX,
                        format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
                        llm=llm,
                        toolkit=st.session_state.toolkit,
                        verbose=True,
                        max_iterations=10,
                        agent_type="openai-tools",
                        top_k=100
                    )
        
        if st.button("Run Query",key=12):
            try:
                response = agent.invoke(chat_query,{"agent_scratchpad": []})['output']
                st.subheader("Response")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        print("None")


with tab2:
    if st.session_state.connected and st.session_state.toolkit and st.session_state.sql_search_tool:
        visual_query = st.text_input("Enter your question:",key=15)
        agent = create_sql_agent(
                        prefix=MSSQL_AGENT_PREFIX,
                        format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
                        llm=llm,
                        toolkit=st.session_state.toolkit,
                        verbose=True,
                        max_iterations=10,
                        agent_type="openai-tools",
                        top_k=100
                    )
        
        if visual_query:
            
            # restructured_query = query_pipeline(visual_query, st.session_state.sql_search_tool, llm)
            
            # st.subheader("Restructured Query")
            # edited_query = st.text_area("Edit the restructured query if needed:", value=restructured_query)

            
            # fig=None
            if st.button("Run Query",key=20):
                try:
                    answer = agent.invoke(visual_query,{"agent_scratchpad": []})['output']
                    print(answer)
                    graph_query=f"Strictly Remember Give me a full plotly python code from the given answer {answer} do not give any explanation all i need is the full python plotly code, include the fig.show() in the ending, strictly do not give any bad code and should not have the string literals or any code issued should not be there and do not give any unterminated string literal errors in the code"
                    response=llm.invoke(graph_query).content
                    code_blocks = re.findall(r'```([\s\S]*?)```', str(response))
                    code=''
                    for code in code_blocks:
                        if code.strip().startswith('python'):
                            code = code.replace('python', '').strip()
                    print(code)
                    # output=[]
                    # exec(code, globals(), {"print": lambda fig: output.append(fig)})
                    # st.subheader("Response")
                    # st.plotly_chart(output[0])
                    exec(code)
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please connect to a database and select at least one table to start chatting.")

with tab3:
    if st.session_state.connected and st.session_state.toolkit and st.session_state.sql_search_tool:

        agent = create_sql_agent(
                        prefix=MSSQL_AGENT_PREFIX,
                        format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
                        llm=llm,
                        toolkit=st.session_state.toolkit,
                        verbose=True,
                        max_iterations=10,
                        agent_type="openai-tools",
                        top_k=100
                    )
        table_query=st.text_input("Input your Table question: ",key=11)
        if table_query:
            if st.button("Run Query: ",key=30):
                try:
                # Invoke the agent with the query
                    process_query_and_display_result(agent,llm,table_query)
            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


with tab4:
    if st.session_state.connected and st.session_state.toolkit and st.session_state.sql_search_tool:
        selected_table = st.selectbox("Select a table", selected_tables)
        if selected_table:
            query = f"SELECT * FROM {selected_table} LIMIT 100"
            df = pd.read_sql_query(query, con=st.session_state.engine)
            st.dataframe(df)
    else:
        st.warning("Please connect to a database and select at least one table to view data.")

with tab5:
    if st.session_state.connected and st.session_state.toolkit and st.session_state.sql_search_tool:
        st.write(find_related_tables_and_columns(selected_tables))


