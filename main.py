import streamlit as st
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
import time

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('words')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def get_similarity_metrics(str1, str2, use_preprocessing=False):
    if use_preprocessing:
        str1 = preprocess_text(str1)
        str2 = preprocess_text(str2)
    
    seq_ratio = SequenceMatcher(None, str1, str2).ratio()
    ratio = fuzz.ratio(str1, str2) / 100
    partial_ratio = fuzz.partial_ratio(str1, str2) / 100
    token_sort = fuzz.token_sort_ratio(str1, str2) / 100
    token_set = fuzz.token_set_ratio(str1, str2) / 100
    
    return {
        "Sequence Matcher": seq_ratio,
        "Simple Ratio": ratio,
        "Partial Ratio": partial_ratio,
        "Token Sort Ratio": token_sort,
        "Token Set Ratio": token_set
    }

def analyze_string_patterns(text):
    char_freq = Counter(text.lower())
    word_freq = Counter(word_tokenize(text.lower()))
    avg_word_len = np.mean([len(word) for word in text.split()])
    
    return {
        'char_frequency': dict(char_freq.most_common(10)),
        'word_frequency': dict(word_freq.most_common(10)),
        'avg_word_length': avg_word_len,
        'total_length': len(text),
        'word_count': len(text.split())
    }

def plot_similarity_distribution(df, metric_cols):
    fig = go.Figure()
    for col in metric_cols:
        fig.add_trace(go.Histogram(x=df[col], name=col, opacity=0.7))
    fig.update_layout(
        title="Similarity Score Distribution",
        xaxis_title="Similarity Score",
        yaxis_title="Count",
        barmode='overlay'
    )
    return fig

def main():
    st.set_page_config(page_title="Advanced Fuzzy Matcher", layout="wide")
    st.title("Enhanced Fuzzy String Matcher")

    # Sidebar controls
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.6, 0.05)
    use_preprocessing = st.sidebar.checkbox("Use Text Preprocessing", True)
    selected_metrics = st.sidebar.multiselect(
        "Select Matching Metrics",
        ["Sequence Matcher", "Simple Ratio", "Partial Ratio", "Token Sort Ratio", "Token Set Ratio"],
        default=["Simple Ratio", "Token Sort Ratio"]
    )

    tabs = st.tabs(["String Comparison", "Bulk Analysis", "Text Analytics"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            string1 = st.text_input("Enter first string:", "example string")
            if string1:
                st.subheader("String 1 Analysis")
                analysis1 = analyze_string_patterns(string1)
                st.json(analysis1)
                
        with col2:
            string2 = st.text_input("Enter second string:", "example text")
            if string2:
                st.subheader("String 2 Analysis")
                analysis2 = analyze_string_patterns(string2)
                st.json(analysis2)

        if string1 and string2:
            metrics = get_similarity_metrics(string1, string2, use_preprocessing)
            
            st.subheader("Similarity Analysis")
            col1, col2 = st.columns(2)
            with col1:
                results_df = pd.DataFrame({
                    'Metric': selected_metrics,
                    'Score': [metrics[m] for m in selected_metrics]
                })
                st.dataframe(results_df.style.background_gradient(cmap='RdYlGn'))
            
            with col2:
                fig = go.Figure([go.Bar(x=selected_metrics, y=[metrics[m] for m in selected_metrics])])
                fig.update_layout(title="Similarity Scores Comparison")
                st.plotly_chart(fig)

    with tabs[1]:
        uploaded_file = st.file_uploader("Upload CSV file with strings to compare", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                source_col = st.selectbox("Select source column", df.columns)
            with col2:
                target_col = st.selectbox("Select target column", df.columns)

            if st.button("Analyze"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    metrics = get_similarity_metrics(
                        str(row[source_col]), 
                        str(row[target_col]),
                        use_preprocessing
                    )
                    result = {
                        'Source': row[source_col],
                        'Target': row[target_col],
                        **{m: metrics[m] for m in selected_metrics}
                    }
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                
                st.subheader("Statistical Summary")
                stats_df = results_df[selected_metrics].describe()
                st.dataframe(stats_df.style.background_gradient(cmap='viridis'))
                
                st.subheader("Distribution Analysis")
                dist_fig = plot_similarity_distribution(results_df, selected_metrics)
                st.plotly_chart(dist_fig)
                
                st.subheader("Correlation Matrix")
                corr_matrix = results_df[selected_metrics].corr()
                fig = px.imshow(corr_matrix)
                st.plotly_chart(fig)
                
                matches_df = results_df[results_df[selected_metrics].mean(axis=1) >= threshold]
                st.write(f"Found {len(matches_df)} matches above threshold {threshold}")
                st.dataframe(matches_df)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "fuzzy_matching_results.csv",
                    "text/csv"
                )

    with tabs[2]:
        text_input = st.text_area("Enter text for analysis", "")
        if text_input:
            analysis = analyze_string_patterns(text_input)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Character Frequency")
                char_fig = px.bar(
                    x=list(analysis['char_frequency'].keys()),
                    y=list(analysis['char_frequency'].values()),
                    title="Character Distribution"
                )
                st.plotly_chart(char_fig)
                
            with col2:
                st.subheader("Word Frequency")
                word_fig = px.bar(
                    x=list(analysis['word_frequency'].keys()),
                    y=list(analysis['word_frequency'].values()),
                    title="Word Distribution"
                )
                st.plotly_chart(word_fig)
            
            st.subheader("Text Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Average Word Length", f"{analysis['avg_word_length']:.2f}")
            with stats_col2:
                st.metric("Total Length", analysis['total_length'])
            with stats_col3:
                st.metric("Word Count", analysis['word_count'])

if __name__ == "__main__":
    main()
