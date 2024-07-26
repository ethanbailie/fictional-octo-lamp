from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from crewai_tools import tool
from dotenv import load_dotenv

import json

load_dotenv()

## set the llm for the agents
llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

## tools
@tool('code generator')
def code_generator(goal: str) -> str:
    """
    Generates code for a given goal and outputs a JSON string
    """

    messages = [
        (
            'system',
            '''
            You are a expert software engineer and are tasked with writing a Python script.
            This script is for a critical function within the company.
            To create the script think it through step-by-step and write the code accordingly.
            You must output a valid Python script with working syntax in a valid JSON string.

            Expected Output:
            {
                'goal': 'some goal',
                'steps': ['step 1: ...', 'step 2: ...'],
                'code': ['x=25', 'y=50', 'return x + y']
            }
            '''
        ),
        (
            'human',
            f'''
            Write a script to accomplish {goal}.
            the output must be in JSON format.
            '''
        )
    ]

    return llm.invoke(messages).content

@tool('code validator')
def code_validator(obj: str) -> str:
    """
    validates code within a given string containing a json object
    """

    messages = [
        (
            'system',
            '''
            You are a expert QA engineer and are tasked with validating a Python script.
            This script is for a critical function within the company.
            To validate the script think it through step-by-step and make suggestions accordingly.
            You must output a valid JSON string.

            Expected Output:
            {
                'goal': 'some goal',
                'steps': ['step 1: ...', 'step 2: ...'],
                'code': ['x=25', 'y="50"', 'return x + y'],
                'suggestions': ['variable y must be an int to add it to the int x']
            }
            '''
        ),
        (
            'human',
            f'''
            Validate the code within {obj} and write down the suggestions.
            the output must be in JSON format.
            '''
        )
    ]

    return llm.invoke(messages).content

@tool('json validator')
def json_validator(obj: str) -> str:
    """
    Validates a JSON string to ensure it is in valid JSON format
    """
    try:
        json.loads(obj)
        return obj
    except Exception as e:
        return f'The JSON string was formatted incorrectly, exception occurred: {e}'
    

## agents
generation_agent = Agent(
    role='Senior software engineer',
    goal='Generate a valid Python script and output it as a JSON string',
    backstory=('You are a senior software engineer entrusted with creating a '
               'critical script for your company.'),
    llm=llm,
    memory=True,
    verbose=True,
    tools=[code_generator]
)

validation_agent = Agent(
    role='Senior QA engineer',
    goal='Validate a JSON string to ensure it is in proper format, and validate the code within and output it as a JSON string',
    backstory=('You are a senior QA engineer entrusted with validating a '
               'critical script for your company.'),
    llm=llm,
    memory=True,
    verbose=True,
    tools=[code_validator, json_validator]
)


## tasks
generation_task = Task(
    description='Generates code according to {goal} and outputs it as a JSON',
    expected_output='A JSON string containing the keys goal, steps, and code',
    agent=generation_agent,
    tools=[code_generator]
)

validation_task = Task(
    description='Validates a JSON string and code within.',
    expected_output='A JSON string containing the keys goal, steps, code, and suggestions',
    agent=validation_agent,
    tools=[code_validator, json_validator]
)


## crew
crew = Crew(
    agents=[generation_agent, validation_agent],
    tasks=[generation_task, validation_task],
    verbose=False
)

## all together
results = crew.kickoff(inputs={'goal': input('Enter your goal here: ')})

print(results)
print('\n')

f = open("full_json.txt", "w")
f.write(results)
f.close()

f = open("generated.py", "w")
for line in json.loads(results)['code']:
    f.write(line)
    f.write('\n')
f.close()