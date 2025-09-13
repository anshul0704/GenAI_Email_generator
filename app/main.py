import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chain import Chain
# from portfolio import Portfolio

def create_streamlit_app(llm):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = loader.load().pop().page_content

            jobs = llm.extract_jobs(data)
            for job in jobs:
                # skills = job.get('skills', [])
                # links = portfolio.query_links(skills)
                email = llm.generate_email(job)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    # portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain)
