from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

def draft_email(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a helpful assistant that drafts an introduction email between company and candidate.
    
    Your goal is to help the user quickly create a perfect email introduction.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Start your reply by saying: "Hi, Here's a draft for your intro:". And then proceed with the reply on a new line.

    Subject: Introducing a Promising Candidate for [Position] at [Company Name]
    Dear [Company Founder's Name],

    
    Make sure to sign of with {signature}.
    
    """

    signature = f"Kind regards, \n\[Name and Lastname]"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email intro draft and consider any other comments from the user for intro as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature )

    return response