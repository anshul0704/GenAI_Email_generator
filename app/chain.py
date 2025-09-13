import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key="YOUR_API_KEY",model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt= PromptTemplate.from_template(
            """
            ### SCRAPPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scrapped text is from a job posting website. 
            Your job is to extract the job postings and return them in JSON format containing the
            following keys: 'role','experience','skills' and 'description'
            Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE)
        """
        )

        chain_extract= prompt | self.llm
        res= chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res= json_parser.parse(res.content)
        except OutputParserException as e:
            print("Error parsing JSON:", e)
            res= {"error": "Unable to parse JSON from the response."}
        return res if isinstance(res, list) else [res]

    def generate_email(self, job):
        prompt= PromptTemplate.from_template(
            """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    Write a personalized and professional cold email to the hiring manager for the above role. 
    Follow these rules:
    1. Subject line must include the **role title** and **company name**.
    2. Address the email to "Hiring Manager" (or the team name if available).
    3. Highlight **relevant skills and achievements with quantifiable impact** 
       (e.g., "improved efficiency by 20%", "led a $5M program").
    4. Clearly show enthusiasm for the company's mission/values from the description.
    5. Keep the tone professional but engaging.
    6. End with a strong call-to-action to discuss further.

    ### OUTPUT:
    A complete email with subject line, greeting, body, and closing.
        """
        )

        chain_email= prompt | self.llm
        res= chain_email.invoke(input={"job_description": job})
        return res.content

if __name__ == "__main__":
    print("Hey")
