from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from toolbox import *

import json

## load env variables for keys
load_dotenv()

## set the llm for the agents
llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

## agents
assessment_agent = Agent(
    role='Goal Assesser',
    goal='Assess the given goal to see if it requires additional context',
    backstory=('You are a senior manager who is experienced with a plethora of requests. '
               'You need to assess if the current goal is able to be completed without any '
               'additional documentation. If an additional document is needed, you '
               'must attempt to retrieve it. If the retrieved results are not good enough, '
               'you must get the user to provide a pdf which you will then embed and retrieve.'),
    llm=llm,
    memory=True,
    verbose=True,
    tools=[assessment, retriever, embed_pdf]
)

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
assessment_task = Task(
    description='Assesses if more context is needed to achieve {goal} and retrieve it if so. If the retrieved information is not enough, prompt to add more context',
    expected_output='A string either of more context or that no additional context is necessary',
    agent=assessment_agent,
    tools=[assessment, retriever, embed_pdf]
)

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
    agents=[generation_agent, validation_agent, assessment_agent],
    tasks=[assessment_task, generation_task, validation_task],
    verbose=False
)

## all together
results = crew.kickoff(inputs={'goal': input('Enter your goal here: ')})

## log results
print(results)
print('\n')

## write full results to text file, write code to a python file
f = open("full_json.txt", "w")
f.write(results)
f.close()

f = open("generated.py", "w")
for line in json.loads(results)['code']:
    f.write(line)
    f.write('\n')
f.close()