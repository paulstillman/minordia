from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from string import Template
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import re

from language_models.ollama_model import OllamaLanguageModel
from retry import retry


system_message = (
    "This is an agent based model. "
    "The goal of the LLM to to play characters in a game, and act as humanlike as possible. "
    "Ideally, human observers should not be able to tell the difference between the LLM and a human player. "
)


def select_action(
    model: OllamaLanguageModel, name, personality, memory, situation, system_message=system_message
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
    output = model.sample_text(request)
    return output


def determine_behaviours(
    model: OllamaLanguageModel, name, personality, memory, situation, system_message=system_message
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
    output = model.sample_text(request)
    return output


def get_outcomes(model: OllamaLanguageModel, actions, personalities, memories, situation):
    """Determines the outcomes of actions using the LLM."""
    outcome_prompts = {}
    for name, action in actions.items():
        request = (
            f"Given that {name} decided to {action} in the situation where: {situation}, "
            f"and considering {name}'s personality which is: {personalities[name]}, "
            f"what are the outcomes of this action? Describe in detail. "
            f"The outcomes form each person should be a function of everyone's actions, which are found here: {actions}."
        )

    outcomes = {
        name: model.sample_text(request)
        for name, prompt in outcome_prompts.items()
    }
    return outcomes


def multiple_choice(
    model: OllamaLanguageModel, options, name, personality, memory, situation, system_message=system_message
):
    """Select an action for an agent based on their personality and memory."""
    request = (
        "This is a multiple choice question. /n"
        f"consider the options: {options} and select the best one for the situation. /n"
        f"Given the situation: {situation}, and the personality: {personality} for {name}, /n"
        f"use the deliberations or memories found in {memory} to help make your decision. /n"
        f"provide only the letter that corresponds to the option that you want to select."
    )
    output = model.sample_text(request)
    return output


def multiple_choice_with_answer(
    model: OllamaLanguageModel, options, name, personality, memory, situation, system_message=system_message
):
    """Select an action for an agent based on their personality and memory."""
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
    output = model.sample_text(request)
    return output


def update_situation(model: OllamaLanguageModel, situation, outcomes):
    """Updates the situation based on LLM-generated outcomes."""
    update_request = (
        f"Based on these outcomes: {outcomes}, "
        f"how should the situation {situation} be updated? Describe the new situation in detail."
    )
    output = model.sample_text(update_request)
    return output


@retry(ValueError, tries=5)
def mental_deliberation(model: OllamaLanguageModel, name: str, personality, memory, situation):
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

    output = model.sample_text(request)
    return output


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
