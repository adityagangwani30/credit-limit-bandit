"""Streamlit dashboard for credit-limit bandit analysis.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st


def main():
    st.set_page_config(
        page_title="Credit-Limit Bandit Dashboard",
        page_icon="💳",
        layout="wide",
    )
    st.title("💳 Credit-Limit Bandit Dashboard")
    st.info("Dashboard under construction – bandit results will appear here.")


if __name__ == "__main__":
    main()
