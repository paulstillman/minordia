from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from string import Template
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import re

from ollama_model import OllamaLanguageModel
from retry import retry


prompt_template = Template(
    """<s>[INST] <<SYS>>
$system_prompt
<</SYS>>

$request 
Answer as if you were the character you are playing. Be as concise as possible. 
Answer:[/INST]"""
)

system_message = (
    "This is an agent based model. "
    "The goal of the LLM to to play characters in a game, and act as humanlike as possible. "
    "Ideally, human observers should not be able to tell the difference between the LLM and a human player. "
)


def select_action(
    llm, name, personality, memory, situation, system_message=system_message
):
    """Select an action for an agent based on their personality and memory."""
    request = (
        f"You need to decide what to do. Pretend that you are {name} and in this situation. "
        f"Given this personality profile: {personality} for {name}, and the current situation: {situation}, "
        f"List 5 actions that you can take in this situation."
        f"You have these memories to help you make a decision: {memory}. "
        f"one you have listed the actions, you will need to select one of them to take."
        f"provide that one action in your response, and have it be less than 100 characters."
    )
    prompt = prompt_template.substitute(system_prompt=system_message, request=request)
    return llm(prompt, stop=["<|eot_id|>"])


def determine_behaviours(
    llm, name, personality, memory, situation, system_message=system_message
):
    """Select a set of appropriate behaviours for an agent based on their personality and memory."""
    request = (
        f"You need to decide what to do. Pretend that you are {name} and in this situation. "
        f"Given this personality profile: {personality} for {name}, and the current situation: {situation}, "
        f"List 5 actions that you can take in this situation."
        f"You have these memories to help you make a decision: {memory}. "
        "Write the answers like a multiple choice question, with the options being the actions that you can take."
        "An example of this would be: (a) action 1, (b) action 2, (c) action 3, (d) action 4, (e) action 5."
    )
    prompt = prompt_template.substitute(system_prompt=system_message, request=request)
    return llm(prompt, stop=["<|eot_id|>"])


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
        outcome_prompts[name] = prompt_template.substitute(
            system_prompt=system_message, request=request
        )
        print()

    outcomes = {
        name: llm(prompt, stop=["<|eot_id|>"])
        for name, prompt in outcome_prompts.items()
    }
    return outcomes


def multiple_choice(
    llm, options, name, personality, memory, situation, system_message=system_message
):
    """Select an action for an agent based on their personality and memory."""
    system_message = (
        "This is an agent based model. /n"
        "The goal of the LLM to to play characters in a game, and act as humanlike as possible. /n"
        "Ideally, human observers should not be able to tell the difference between the LLM and a human player. /n"
        f"You are playing a character named {name}. /n"
        f"Pretend that you are {name} and answer the question as if you were them. /n"
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
    return llm(prompt, stop=["<|eot_id|>"])


def multiple_choice_with_answer(
    llm, options, name, personality, memory, situation, system_message=system_message
):
    """Select an action for an agent based on their personality and memory."""
    system_message = (
        f"You are playing a character named {name}. /n"
        f"Pretend that you are {name} and answer the question as if you were them. /n"
        "This is in the context of a role playing game. /n"
        "You must select the best option for the situation. /n"
    )
    request = (
        "This is a multiple choice question. /n"
        f"consider the options: {options} and select the best one for the situation. /n"
        f"Given the situation: {situation}, and the personality: {personality} for {name}, /n"
        f"use the deliberations or memories found in {memory} to help make your decision. /n"
        f"provide only the letter that corresponds to the option that you want to select."
        "and the the behaviour that it corresponds to. /n"
        "when listing the behaviour, make sure that it is word for word the same as the option that the letter is associated with."
    )
    prompt = prompt_template.substitute(system_prompt=system_message, request=request)
    return llm(prompt, stop=["<|eot_id|>"])


def update_situation(llm, situation, outcomes):
    """Updates the situation based on LLM-generated outcomes."""
    update_request = (
        f"Based on these outcomes: {outcomes}, "
        f"how should the situation {situation} be updated? Describe the new situation in detail."
    )
    prompt = prompt_template.substitute(
        system_prompt=system_message, request=update_request
    )
    new_situation = llm(prompt, stop=["<|eot_id|>"])
    return new_situation


def mental_deliberation(
    llm, name, personality, memory, situation, system_message=system_message
):
    """Mental deliberation for an agent based on their personality and memory."""
    request = (
        f"Given this personality profile: {personality} for {name}, and the current situation: {situation}, "
        f"what insights can you gain from your memories: {memory}? "
        f"how should this inform your decision making process? "
        f"you should only use information that {name} would have access to or would have experienced."
        f"do not use information that is not relevant to the situation or that would not be known to {name}."
        f"use more recent rounds to find your plan, and try to figire out how the sequence of actions and rewards will play out. "
        f"recent rounds are the ones with a higher round number. "
        f"sometimes a random action can be a good idea to discover new policies, but not too often. "
        f"remember if you tried the random action recently so you don't do it too often"
    )

    prompt = prompt_template.substitute(system_prompt=system_message, request=request)
    return llm(prompt, stop=["<|eot_id|>"])


def extract_mcqs(text):
    # Regular expression to find patterns like "(a) option text"
    pattern = re.compile(r"\([a-z]\) [^\)]+")

    # Find all occurrences of the pattern
    mcqs = pattern.findall(text)

    # Join all the extracted MCQs into a single string separated by new lines
    result = "\n".join(mcqs)
    return result


## Ollama models


@retry(ValueError, tries=5)
def compute_desire_for_gamble(model: OllamaLanguageModel, object: str):
    """compute value."""
    request = (
        f"You are very logical and rational when doing this task"
        f"You are presented with a gamble. it has a probability of winning, a value of winning, and a value of losing. "
        f"If you win, you get the win value, if you lose, you get loss value. "
        f"The probability of winning is the 'win_probability']. "
        f"You need to think about an option, and how desirable it is. "
        f"Compute the expected value of the gamble first. "
        f"Think about how good or bad it is and provide a affective feeling preference value between -1 and 1 "
        f"which corresponds to the desirability of the option. "
        f"Use -1 for very bad, 0 for neutral, and 1 for very good. "
        f"The option is: {object}"
        f"Provide this answer in the form of a number between -1 and 1. "
        f"Provide only a single number as the response."
        f"Do not provide any explanations, just provide a single number."
    )

    output = model.sample_text(request)
    return float(output)
