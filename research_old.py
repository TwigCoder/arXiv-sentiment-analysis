import streamlit as st
import pandas as pd
import numpy as np
import arxiv
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime, timedelta
import altair as alt
from collections import defaultdict
import time
from typing import List, Dict, Tuple
import logging
import warnings


class ArxivSentimentAnalyzer:
    def __init__(self):
        self.cached_results = {}
        self.client = arxiv.Client(page_size=100, delay_seconds=1, num_retries=3)

    def get_papers(
        self, query: str, max_results: int = 100, date_filter: str = None
    ) -> List[Dict]:
        cache_key = f"{query}_{max_results}_{date_filter}"
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]

        try:
            if date_filter:
                year, month = date_filter.split("-")
                query = (
                    f"{query} AND submittedDate:[{year}{month}01 TO {year}{month}31]"
                )

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            papers = []
            for result in self.client.results(search):
                papers.append(
                    {
                        "title": result.title,
                        "abstract": result.summary,
                        "date": result.published,
                        "authors": [author.name for author in result.authors],
                        "categories": [cat for cat in result.categories],
                        "url": result.entry_id,
                    }
                )

            self.cached_results[cache_key] = papers
            return papers

        except Exception as e:
            st.error(f"Error fetching papers: {str(e)}")
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        try:
            analysis = TextBlob(text)
            return {
                "polarity": analysis.sentiment.polarity,
                "subjectivity": analysis.sentiment.subjectivity,
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"polarity": 0.0, "subjectivity": 0.0}

    def process_papers(self, papers: List[Dict]) -> pd.DataFrame:
        data = []
        for paper in papers:
            sentiment = self.analyze_sentiment(paper["abstract"])
            data.append(
                {
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "date": paper["date"],
                    "polarity": sentiment["polarity"],
                    "subjectivity": sentiment["subjectivity"],
                    "category": (
                        paper["categories"][0] if paper["categories"] else "Unknown"
                    ),
                    "url": paper["url"],
                    "authors": ", ".join(paper["authors"]),
                }
            )
        return pd.DataFrame(data)


class DashboardUI:
    def __init__(self):
        self.analyzer = ArxivSentimentAnalyzer()
        if "comparison_data" not in st.session_state:
            st.session_state.comparison_data = None

    def setup_page(self):
        st.set_page_config(
            page_title="arXiv Sentiment Analyzer", page_icon="📚", layout="wide"
        )
        st.title("arXiv Research Sentiment Analyzer")

    def create_sidebar(self) -> Tuple[str, int, str, str, str]:
        with st.sidebar:
            st.header("Search Parameters")
            tab1, tab2 = st.tabs(["Main Search", "Comparison"])

            with tab1:
                query = st.text_input("Search Query", value="artificial intelligence")
                max_results = st.slider("Maximum Papers", 10, 200, 50)
                date_filter = st.date_input("Filter by Date", None)
                date_str = date_filter.strftime("%Y-%m") if date_filter else None
                chart_color = st.color_picker("Chart Color", "#1f77b4")
                sort_by = st.radio(
                    "Sort Results By", ["Date", "Polarity", "Subjectivity"]
                )

            with tab2:
                comparison_query = st.text_input(
                    "Comparison Query", value="machine learning"
                )
                if st.button("Run Comparison"):
                    with st.spinner("Fetching comparison data..."):
                        comparison_papers = self.analyzer.get_papers(
                            comparison_query, max_results
                        )
                        if comparison_papers:
                            st.session_state.comparison_data = (
                                self.analyzer.process_papers(comparison_papers)
                            )

            return query, max_results, date_str, chart_color, sort_by

    def plot_sentiment_trends(self, df: pd.DataFrame, chart_color: str):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Polarity Over Time")
            base_chart = (
                alt.Chart(df)
                .mark_circle()
                .encode(
                    x="date:T",
                    y="polarity:Q",
                    color=alt.value(chart_color),
                    tooltip=["title", "polarity", "date", "category"],
                )
                .interactive()
            )

            if st.session_state.comparison_data is not None:
                comparison_chart = (
                    alt.Chart(st.session_state.comparison_data)
                    .mark_circle(opacity=0.5)
                    .encode(
                        x="date:T",
                        y="polarity:Q",
                        color=alt.value("red"),
                        tooltip=["title", "polarity", "date", "category"],
                    )
                )
                chart = base_chart + comparison_chart
            else:
                chart = base_chart

            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.subheader("Subjectivity Distribution by Category")
            fig = px.box(
                df,
                x="category",
                y="subjectivity",
                points="all",
                color_discrete_sequence=[chart_color],
            )
            st.plotly_chart(fig, use_container_width=True)

    def display_paper_table(self, df: pd.DataFrame):
        st.subheader("Paper Details")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search within results", "")
        with col2:
            filter_polarity = st.select_slider(
                "Filter by Polarity",
                options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                value=(-1.0, 1.0),
            )

        display_df = df.copy()
        if search_term:
            mask = display_df["title"].str.contains(
                search_term, case=False
            ) | display_df["abstract"].str.contains(search_term, case=False)
            display_df = display_df[mask]

        display_df = display_df[
            (display_df["polarity"] >= filter_polarity[0])
            & (display_df["polarity"] <= filter_polarity[1])
        ]

        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df = display_df.sort_values("date", ascending=False)

        st.dataframe(
            display_df[
                [
                    "title",
                    "date",
                    "polarity",
                    "subjectivity",
                    "category",
                    "url",
                    "authors",
                ]
            ],
            column_config={"url": st.column_config.LinkColumn("Paper Link")},
            hide_index=True,
        )

    def display_stats(self, df: pd.DataFrame):
        st.subheader("Statistics Dashboard")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Polarity", f"{df['polarity'].mean():.3f}")
        with col2:
            st.metric("Average Subjectivity", f"{df['subjectivity'].mean():.3f}")
        with col3:
            st.metric("Total Papers", len(df))

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Most Common Category", df["category"].mode()[0])
        with col5:
            st.metric("Unique Authors", len(set(",".join(df["authors"]).split(", "))))
        with col6:
            current_time = pd.Timestamp.now(tz="UTC")
            recent_papers = df[df["date"] >= current_time - pd.Timedelta(days=30)]
            st.metric("Papers Last 30 Days", len(recent_papers))

    def display_paper_table(self, df: pd.DataFrame):
        st.subheader("Paper Details")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search within results", "")
        with col2:
            filter_polarity = st.select_slider(
                "Filter by Polarity",
                options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                value=(-1.0, 1.0),
            )

        display_df = df.copy()
        if search_term:
            mask = display_df["title"].str.contains(
                search_term, case=False
            ) | display_df["abstract"].str.contains(search_term, case=False)
            display_df = display_df[mask]

        display_df = display_df[
            (display_df["polarity"] >= filter_polarity[0])
            & (display_df["polarity"] <= filter_polarity[1])
        ]

        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df = display_df.sort_values("date", ascending=False)

        st.dataframe(
            display_df[
                [
                    "title",
                    "date",
                    "polarity",
                    "subjectivity",
                    "category",
                    "url",
                    "authors",
                ]
            ],
            column_config={"url": st.column_config.LinkColumn("Paper Link")},
            hide_index=True,
        )

    def display_topic_analysis(self, df: pd.DataFrame):
        st.subheader("Topic Analysis")

        categories = df["category"].unique()
        selected_categories = st.multiselect(
            "Select Categories to Compare", categories, default=categories[:2]
        )

        if len(selected_categories) > 0:
            filtered_df = df[df["category"].isin(selected_categories)]

            fig = go.Figure()
            for category in selected_categories:
                cat_data = filtered_df[filtered_df["category"] == category]
                fig.add_trace(
                    go.Violin(
                        y=cat_data["polarity"],
                        name=category,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )

            fig.update_layout(title="Sentiment Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        self.setup_page()
        query, max_results, date_str, chart_color, sort_by = self.create_sidebar()

        with st.spinner("Fetching and analyzing papers..."):
            papers = self.analyzer.get_papers(query, max_results, date_str)
            if papers:
                df = self.analyzer.process_papers(papers)
                df["date"] = pd.to_datetime(df["date"])

                self.display_stats(df)
                self.plot_sentiment_trends(df, chart_color)
                self.display_topic_analysis(df)
                self.display_paper_table(df)

                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Data", csv, "arxiv_sentiment_analysis.csv", "text/csv"
                )
            else:
                st.error("No papers found. Please try a different search query.")


if __name__ == "__main__":
    app = DashboardUI()
    app.run()
