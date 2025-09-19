import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import httpx
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datetime import datetime

# Load environment variables
load_dotenv()

# === Streamlit Page Config and Custom Theme ===
st.set_page_config(page_title="Engineering Logs Interpreter", layout="wide")
st.markdown("""
<style>
body {background-color: #121212; color: #E0E0E0;}
[data-testid="stSidebar"] {background-color: #87CEFA; color: white; width: 35%;}
h1,h2,h3,h4,h5 {color: #E0E0E0;}
.chat-box {
    background-color: #121212;
    color: white;
    padding: 10px;
    border-radius: 10px;
    max-height: 60vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
    gap: 10px;
}
.user-msg, .assistant-msg {
    padding: 10px 15px;
    border-radius: 16px;
    max-width: 60%;
    word-wrap: break-word;
    position: relative;
}
.user-msg {
    background-color: #ADD8E6;
    color: black;
    text-align: right;
    border-bottom-right-radius: 0;
}
.assistant-msg {
    background-color: #1E3A8A;
    color: white;
    text-align: left;
    border-bottom-left-radius: 0;
}
.msg-meta {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-top: 5px;
}
.chat-entry {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    margin-bottom: 8px;
}
.user-entry {
    justify-content: flex-end;
    flex-direction: row-reverse;
}
.assistant-entry {
    justify-content: flex-start;
    flex-direction: row;
}
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: #2E2E2E;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.recommendation-box, .anomalies-box {
    background-color: #2A2A2A;
    color: #E0E0E0;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}
[data-testid="stFileUploader"] button {
    background-color: #87CEFA;
    color: black;
    border-radius: 8px;
    padding: 5px 12px;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #cbbf96;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# === Sidebar: Upload + User Type ===
with st.sidebar:
    st.markdown("<h1 style='font-size:30px;'>Engineering Logs Interpreter</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    user_type = st.selectbox("Select User Type", ["Project Manager", "Developer"])

# === Functions ===
def analyze_data(file):
    data = pd.read_csv(file)
    required_cols = {'timestamp', 'label', 'response_time_ms', 'status', 'thread', 'bytes_sent', 'bytes_received', 'throughput_rps', 'cpu_usage_pct', 'memory_usage_mb'}
    if not required_cols.issubset(set(data.columns)):
        missing = required_cols - set(data.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    features = data[['response_time_ms', 'status', 'bytes_sent', 'bytes_received']]
    model = IsolationForest(contamination=0.1, random_state=42)
    data['anomaly'] = model.fit_predict(features)
    data['is_anomaly'] = data['anomaly'] == -1

    total_requests = len(data)
    error_rate = (data['status'] >= 400).mean() * 100
    avg_response_time = data['response_time_ms'].mean()

    grouped = data.groupby('label').agg({
        'response_time_ms': ['mean', lambda x: np.percentile(x, 95)],
        'status': lambda x: (x >= 400).mean() * 100,
        'is_anomaly': 'sum',
        'throughput_rps': 'mean',
        'cpu_usage_pct': 'mean',
        'memory_usage_mb': 'mean',
        'label': 'count'
    }).reset_index()

    grouped.columns = ['Endpoint', 'Avg RT (ms)', 'P95 RT (ms)', 'Error %', 'Anomalies',
                       'Avg Throughput (rps)', 'Avg CPU (%)', 'Avg Mem (MB)', 'Endpoint Count']

    grouped['Availability %'] = 100 - grouped['Error %']
    grouped['Reliability %'] = 100 - (grouped['Anomalies'] / grouped['Anomalies'].sum() * 100).fillna(0)

    anomalies_list = data[data['is_anomaly']][['timestamp','label','response_time_ms','status']].to_dict('records')

    return total_requests, avg_response_time, error_rate, grouped, anomalies_list, data

def init_llm():
    base_url = os.getenv("api_endpoint")
    api_key = os.getenv("api_key")
    model_file = "models.txt"
    try:
        with open(model_file, "r") as f:
            model_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        model_names = ["gpt-4o-mini"]
    client = httpx.Client(verify=False)
    llm_instances = []
    for model_name in model_names:
        llm = ChatOpenAI(base_url=base_url, model=model_name, api_key=api_key, http_client=client)
        llm_instances.append((model_name, llm))
    return llm_instances

def ask_agent(user_input, llm_instances, summary_string):
    if llm_instances:
        _, llm = llm_instances[0]
        prompt = (
            f"You have the following load test summary and per-endpoint metrics:\n\n"
            f"{summary_string}\n\n"
            f"Answer the user question concisely. Only provide the answer.\n\n"
            f"Do not assume information not present in the logs. "
            f"Avoid technical jargon unless the user explicitly asks for it. "
            f"If a question cannot be answered from the data, respond: 'The provided logs do not contain this information.' "
            f"If a question is not relevant to the data provided, respond: 'Apologies, I am unable to answer this question as this is not relevant to this use case.'\n\n"
            f"User question: {user_input}"
        )
        try:
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as e:
            return f"Error from LLM: {e}"
    else:
        return "No LLM available."

def get_llm_recommendations(llm_instances, summary_string):
    if llm_instances:
        _, llm = llm_instances[0]
        prompt = (
            f"You are analyzing load test results with the following endpoint-level performance summary:\n\n"
            f"{summary_string}\n\n"
            f"Based on anomalies, error rates, response times, and resource usage, suggest 2-4 specific and actionable recommendations to improve performance, stability, or reliability. "
            f"Be concise, clear, and avoid repeating the metrics themselves."
        )
        try:
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as e:
            return [f"Error generating recommendations from LLM: {e}"]
    else:
        return ["No LLM available for recommendations."]

# === Main Panel ===
st.markdown("<h2 style='text-align:center;'>Load Test Logs Analysis</h2>", unsafe_allow_html=True)

if uploaded_file:
    try:
        total_requests, avg_rt, error_rate, grouped, anomalies_list, raw_data = analyze_data(uploaded_file)

        st.markdown(f"**Total Requests:** {total_requests} | **Avg Response Time:** {avg_rt:.1f} ms | **Error Rate:** {error_rate:.1f}%")

        st.markdown("### Per-Endpoint Summary")
        st.dataframe(grouped)

        st.markdown("### Anomalies Detected")
        if anomalies_list:
            st.markdown("<div class='anomalies-box'>", unsafe_allow_html=True)
            for a in anomalies_list:
                st.markdown(f"• {a['label']} at {a['timestamp']} (RT: {a['response_time_ms']} ms, Status: {a['status']})", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='anomalies-box'>✅ No anomalies detected</div>", unsafe_allow_html=True)

        # Initialize LLM
        if 'llm_instances' not in st.session_state:
            st.session_state.llm_instances = init_llm()

        summary_string = grouped.to_string(index=False)

        # === LLM Recommendations ===
        st.markdown("### Recommendations")
        rec_text = get_llm_recommendations(st.session_state.llm_instances, summary_string)
        st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
        for line in rec_text.split("\n"):
            if line.strip():
                st.markdown(f"• {line.strip()}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # === Chat Interface ===
        st.markdown("### Ask the Engineering Logs Interpreter")
        user_input = st.text_input("Type your question here...", key="chat_input_main")

        if st.button("Send Question"):
            if user_input.strip():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                answer = ask_agent(user_input, st.session_state.llm_instances, summary_string)
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({
                    'question': user_input,
                    'answer': answer,
                    'timestamp': timestamp
                })

        # === Chat History (Updated loop) ===
        st.markdown("### Chat History")
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

        for chat in reversed(st.session_state.get('chat_history', [])):
            # User message
            st.markdown(f"""
                <div class='chat-entry user-entry'>
                    <div class='avatar'>👤</div>
                    <div class='user-msg'>
                        {chat['question']}
                        <div class='msg-meta'>{chat['timestamp']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Assistant message
            st.markdown(f"""
                <div class='chat-entry assistant-entry'>
                    <div class='avatar'>🤖</div>
                    <div class='assistant-msg'>
                        {chat['answer']}
                        <div class='msg-meta'>{chat['timestamp']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.markdown("<h4 style='text-align:center; color:#E0E0E0;'>Please upload a CSV file to start the analysis</h4>", unsafe_allow_html=True)
