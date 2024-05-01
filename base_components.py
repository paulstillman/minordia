from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from string import Template
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

prompt_template = Template("""<s>[INST] <<SYS>>
$system_prompt
<</SYS>>

$request 
Answer as if you were the character you are playing. Be as concise as possible. 
Answer:[/INST]""")

system_message = ("This is an agent based model. "
    f"The goal of the LLM to to play the characters. "
    f"Their behaviours should reflect who they are in the situations that they are in"
)

def select_action(llm, name, personality, memory, situation, system_message=system_message):
    """Select an action for an agent based on their personality and memory."""
    request = (
        "You need to decide what to do. Pretend that you are {name} and in this situation. "
        f"Given this personality profile: {personality} for {name}, and the current situation: {situation}, "
        f"List 5 actions that you can take in this situation."
        f"You have these memories to help you make a decision: {memory}. "
        f"one you have listed the actions, you will need to select one of them to take."
        f"provide that one action in your response, and have it be less than 100 characters."
    )
    prompt = prompt_template.substitute(system_prompt=system_message, request=request)
    return llm(prompt, stop=['<|eot_id|>'])

def get_outcomes(llm, actions, personalities, memories, situation):
    """Determines the outcomes of actions using the LLM."""
    outcome_prompts = {}
    for name, action in actions.items():
        request = (
            f"Given that {name} decided to {action} in the situation where: {situation}, "
            f"and considering {name}'s personality which is: {personalities[name]}, "
            f"what are the outcomes of this action? Describe in detail. "
            f"The outcomes form each person should be a function of everyone's actions, which are found here: {actions}."
        )
        outcome_prompts[name] = prompt_template.substitute(system_prompt=system_message, request=request)
        print()
    
    outcomes = {name: llm(prompt, stop=['<|eot_id|>']) for name, prompt in outcome_prompts.items()}
    return outcomes

def multiple_choice(llm, options, name, personality, memory, situation, system_message=system_message):
    """Select an action for an agent based on their personality and memory."""
    system_message = (
        "You are playing a character named {name}. /n"
        "Pretend that you are {name} and answer the question as if you were them. /n"
        "This is in the context of a role playing game. /n"
        "You must select the best option for the situation. /n"
        "You must only provide the letter that corresponds to the option that you want to select. /n"
    )
    request = (
        "This is a multiple choice question. /n"
        f"consider the options: {options} and select the best one for the situation. /n"
        f"Given the situation: {situation}, and the personality: {personality} for {name}, /n"
        f"use the deliberations or memories found in {memory} to help make your decision. /n"
        f"provide only the letter that corresponds to the option that you want to select."
    )
    prompt = prompt_template.substitute(system_prompt=system_message, request=request)
    return llm(prompt, stop=['<|eot_id|>'])

def update_situation(llm, situation, outcomes):
    """Updates the situation based on LLM-generated outcomes."""
    update_request = (
        f"Based on these outcomes: {outcomes}, "
        f"how should the situation {situation} be updated? Describe the new situation in detail."
    )
    prompt = prompt_template.substitute(system_prompt=system_message, request=update_request)
    new_situation = llm(prompt, stop=['<|eot_id|>'])
    return new_situation

def mental_deliberation(llm, name, personality, memory, situation, system_message=system_message):
    """Mental deliberation for an agent based on their personality and memory."""
    request = (
        f"Given this personality profile: {personality} for {name}, and the current situation: {situation}, "
        f"what insights can you gain from your memories: {memory}? "
        f"how should this inform your decision making process? "
        f"use more recent rounds to find your plan, and try to figire out how the sequence of actions and rewards will play out. "
        f"recent rounds are the ones with a higher round number. "
        f"sometimes a random action can be a good idea to discover new policies, but not too often. "
        f"remember if you tried the random action recently so you don't do it too often"
    )

    prompt = prompt_template.substitute(system_prompt=system_message, request=request)
    return llm(prompt, stop=['<|eot_id|>'])