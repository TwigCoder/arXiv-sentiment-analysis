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

    def create_sidebar(self) -> Tuple[str, int, str, str, str, str]:
        with st.sidebar:
            st.header("Search Parameters")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Main Search")
                query = st.text_input("Query", value="artificial intelligence")
            with col2:
                st.subheader("Comparison")
                comparison_query = st.text_input("Query 2", value="machine learning")

            max_results = st.slider("Maximum Papers per Query", 10, 200, 50)
            date_str = None
            chart_color = st.color_picker("Chart Color", "#1f77b4")
            sort_by = "Date"

            return query, max_results, date_str, chart_color, sort_by, comparison_query

    def plot_sentiment_trends(
        self, df: pd.DataFrame, comparison_df: pd.DataFrame, chart_color: str
    ):
        st.subheader("Time Series Analysis")

        col1, col2 = st.columns(2)
        with col1:
            base_chart = (
                alt.Chart(df)
                .mark_circle()
                .encode(
                    x="date:T",
                    y="polarity:Q",
                    color=alt.value(chart_color),
                    tooltip=["title", "polarity", "date"],
                )
                .properties(title="Main Query - Sentiment Over Time")
                .interactive()
            )
            st.altair_chart(base_chart, use_container_width=True)

        with col2:
            comparison_chart = (
                alt.Chart(comparison_df)
                .mark_circle()
                .encode(
                    x="date:T",
                    y="polarity:Q",
                    color=alt.value("red"),
                    tooltip=["title", "polarity", "date"],
                )
                .properties(title="Comparison Query - Sentiment Over Time")
                .interactive()
            )
            st.altair_chart(comparison_chart, use_container_width=True)

        st.subheader("Category Analysis")
        col3, col4 = st.columns(2)
        with col3:
            fig1 = px.box(
                df,
                x="category",
                y="subjectivity",
                title="Main Query - Subjectivity by Category",
                color_discrete_sequence=[chart_color],
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col4:
            fig2 = px.box(
                comparison_df,
                x="category",
                y="subjectivity",
                title="Comparison Query - Subjectivity by Category",
                color_discrete_sequence=["red"],
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Topic Analysis")
        col5, col6 = st.columns(2)

        with col5:
            categories = df["category"].unique()
            selected_categories = st.multiselect(
                "Select Categories (Main Query)", categories, default=categories[:2]
            )
            if selected_categories:
                filtered_df = df[df["category"].isin(selected_categories)]
                fig3 = go.Figure()
                for category in selected_categories:
                    cat_data = filtered_df[filtered_df["category"] == category]
                    fig3.add_trace(
                        go.Violin(
                            y=cat_data["polarity"],
                            name=category,
                            box_visible=True,
                            meanline_visible=True,
                        )
                    )
                fig3.update_layout(
                    title="Main Query - Sentiment Distribution by Category"
                )
                st.plotly_chart(fig3, use_container_width=True)

        with col6:
            comp_categories = comparison_df["category"].unique()
            selected_comp_categories = st.multiselect(
                "Select Categories (Comparison Query)",
                comp_categories,
                default=comp_categories[:2],
            )
            if selected_comp_categories:
                filtered_comp_df = comparison_df[
                    comparison_df["category"].isin(selected_comp_categories)
                ]
                fig4 = go.Figure()
                for category in selected_comp_categories:
                    cat_data = filtered_comp_df[
                        filtered_comp_df["category"] == category
                    ]
                    fig4.add_trace(
                        go.Violin(
                            y=cat_data["polarity"],
                            name=category,
                            box_visible=True,
                            meanline_visible=True,
                        )
                    )
                fig4.update_layout(
                    title="Comparison Query - Sentiment Distribution by Category"
                )
                st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Sentiment vs Subjectivity")
        col7, col8 = st.columns(2)

        with col7:
            fig5 = px.scatter(
                df,
                x="subjectivity",
                y="polarity",
                color="category",
                title="Main Query - Sentiment vs Subjectivity",
                hover_data=["title"],
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col8:
            fig6 = px.scatter(
                comparison_df,
                x="subjectivity",
                y="polarity",
                color="category",
                title="Comparison Query - Sentiment vs Subjectivity",
                hover_data=["title"],
            )
            st.plotly_chart(fig6, use_container_width=True)

    def display_paper_table(self, df: pd.DataFrame, table_key: str):
        st.subheader("Paper Details")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input(
                "Search within results", "", key=f"search_{table_key}"
            )
        with col2:
            filter_polarity = st.select_slider(
                "Filter by Polarity",
                options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                value=(-1.0, 1.0),
                key=f"polarity_{table_key}",
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
            key=f"dataframe_{table_key}",
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

    def display_stats_comparison(self, df: pd.DataFrame, comparison_df: pd.DataFrame):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Main Query Stats")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Average Polarity", f"{df['polarity'].mean():.3f}")
                st.metric("Total Papers", len(df))
            with stats_col2:
                st.metric("Average Subjectivity", f"{df['subjectivity'].mean():.3f}")
                st.metric(
                    "Unique Authors", len(set(",".join(df["authors"]).split(", ")))
                )

        with col2:
            st.subheader("Comparison Query Stats")
            stats_col3, stats_col4 = st.columns(2)
            with stats_col3:
                st.metric("Average Polarity", f"{comparison_df['polarity'].mean():.3f}")
                st.metric("Total Papers", len(comparison_df))
            with stats_col4:
                st.metric(
                    "Average Subjectivity",
                    f"{comparison_df['subjectivity'].mean():.3f}",
                )
                st.metric(
                    "Unique Authors",
                    len(set(",".join(comparison_df["authors"]).split(", "))),
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
        query, max_results, date_str, chart_color, sort_by, comparison_query = (
            self.create_sidebar()
        )

        with st.spinner("Fetching papers..."):
            papers = self.analyzer.get_papers(query, max_results, date_str)
            comparison_papers = self.analyzer.get_papers(
                comparison_query, max_results, date_str
            )

            if papers and comparison_papers:
                df = self.analyzer.process_papers(papers)
                comparison_df = self.analyzer.process_papers(comparison_papers)

                df["date"] = pd.to_datetime(df["date"])
                comparison_df["date"] = pd.to_datetime(comparison_df["date"])

                self.display_stats_comparison(df, comparison_df)
                self.plot_sentiment_trends(df, comparison_df, chart_color)

                tab1, tab2 = st.tabs(["Main Query Results", "Comparison Query Results"])
                with tab1:
                    self.display_paper_table(df, "main")
                with tab2:
                    self.display_paper_table(comparison_df, "comparison")

                csv = pd.concat([df, comparison_df]).to_csv(index=False)
                st.download_button(
                    "Download All Data", csv, "arxiv_sentiment_analysis.csv", "text/csv"
                )
            else:
                st.error(
                    "No papers found for one or both queries. Please try different search terms."
                )


if __name__ == "__main__":
    app = DashboardUI()
    app.run()
