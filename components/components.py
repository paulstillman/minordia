from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from string import Template
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import re

from scipy.special import softmax

from language_models.ollama_model import OllamaLanguageModel
from retry import retry


system_message = (
    "This is an agent based model. "
    "The goal of the LLM to to play characters in a game, and act as humanlike as possible. "
    "Ideally, human observers should not be able to tell the difference between the LLM and a human player. "
)


def select_action(
    model: OllamaLanguageModel,
    name,
    personality,
    memory,
    situation,
    system_message=system_message,
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
    model: OllamaLanguageModel,
    name,
    personality,
    memory,
    situation,
    system_message=system_message,
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


def get_outcomes(
    model: OllamaLanguageModel, actions, personalities, memories, situation
):
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
        name: model.sample_text(request) for name, prompt in outcome_prompts.items()
    }
    return outcomes


@retry(AttributeError, tries=5)
def multiple_choice(
    model: OllamaLanguageModel,
    options,
    name,
    personality,
    memory,
    situation,
    system_message=system_message,
):
    """Select an action for an agent based on their personality and memory."""
    request = (
        "This is a multiple choice question. /n"
        f"consider the options: {options} and select the best one for the situation. /n"
        f"Given the situation: {situation}, and the personality: {personality} for {name}, /n"
        f"use the deliberations or memories found in {memory} to help make your decision. /n"
        f"provide only the letter that corresponds to the option that you want to select."
        f"example: (a)"
    )
    output = model.sample_text(request)

    if len(output) > 1:
        try:
            output = re.search(r"\(?(\w)(?=\))", output).group(1)
        except ValueError as e:
            print("No match found", e)

    return output


def multiple_choice_with_answer(
    model: OllamaLanguageModel,
    options,
    name,
    personality,
    memory,
    situation,
    system_message=system_message,
):
    """Select an action for an agent based on their personality and memory."""
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
def mental_deliberation(
    model: OllamaLanguageModel, name: str, personality, memory, situation
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


@retry(ValueError, tries=5)
def compute_distribution_of_desire_for_gamble(
    model: OllamaLanguageModel, object: str, options=None
):
    """compute value."""
    if options is None:
        max_value = 5
        min_value = -5
        options = list(map(str, range(min_value, max_value + 1)))

    request = (
        f"You are very logical and rational when doing this task"
        f"You are presented with a gamble. it has a probability of winning, a value of winning, and a value of losing. "
        f"If you win, you get the win value, if you lose, you get loss value. "
        f"The probability of winning is the 'win_probability']. "
        f"You need to think about an option, and how desirable it is. "
        f"Think about how good or bad it is and provide a affective feeling preference value between {min_value} and {max_value} "
        f"which corresponds to the value of the taking the gamble or object. "
        f"The option is: {object}"
        f"Provode the probability of each option being selected. "
        f"In other words, provide a probability for each of the {options}"
        f"Provide these probabilities in the form of a list of numbers that sum to 1. "
        f"Provide only the numbers as the response."
        f"Do not provide any explanations, just provide a list of numbers."
    )

    for attempt in range(5):
        output = model.sample_text(request)

        # Parse the output to extract the list of numbers
        numbers_str = output.strip().split(",")
        numbers_str = [num.strip() for num in numbers_str if num.strip()]
        numbers = [float(num) for num in numbers_str]

        if len(numbers) != len(options):
            continue  # Retry if the number of probabilities doesn't match the number of options

        # Softmax the numbers
        softmaxed_probs = np.exp(numbers) / np.sum(np.exp(numbers))

        # Multiply softmaxed probabilities by options to get expected value
        expected_value = np.sum(np.array(options, dtype=float) * softmaxed_probs)

        return output, expected_value

    raise ValueError("Failed to compute distribution after maximum attempts.")


def multiple_choice_preferences(
    model: OllamaLanguageModel, options: None, gamble, affective_feeling
):
    """Select an action for an agent based on their personality and memory."""
    if options is None:
        options = ["Never", "Rarely", "Sometimes", "Often", "Always"]
    request = (
        f"Consider these options: {options} and select the best one for the situation. "
        f"Given this gamble: {gamble}, and this affetive feeling about the gamble: {affective_feeling}, "
        f"Which {options} best characterizes the likelihood that you would take this this gamble? "
        f"Provide only the option that corresponds to your likelihood to take the gamble."
        f"Do not provide any explanations, just provide a single word."
    )
    output = model.sample_text(request)
    return output


def compute_distribution_of_desire_for_gamble_binaries(model, object_str, options=None):
    """Compute the distribution of desire for a gamble."""

    if options is None:
        max_value = 5
        min_value = -5
        options = list(map(str, range(min_value, max_value + 1)))
    else:
        options = [str(option) for option in options]  # Convert options to strings

    original_probabilities = []
    for option in options:
        request = (
            f"You are very logical and rational when doing this task. "
            f"You are thinking about your responses to this gamble: {object_str}. "
            f"Consider the answer '{option}' within the context of the entire range of options: {options}. "
            f"How do you feel about this option relative to the others? "
            f"Please provide a numerical rating between 0 and 1, "
            f"where 0 indicates much less desirable than the others "
            f"and 1 indicates much more desirable than the others. "
            f"It will be helpful to first consider you response to the all the options and then think about this one relative to all the others."
        )
        output = model.sample_text(request)
        try:
            rating = float(output)
            # Normalize the rating to be between 0 and 1
            # normalized_rating = (rating - min_value) / (max_value - min_value)
            original_probabilities.append(rating)
        except ValueError:
            # Handle invalid rating responses
            original_probabilities.append(0.0)

    # Apply softmax to original probabilities
    softmax_probabilities = softmax(original_probabilities)

    return original_probabilities, softmax_probabilities


def compute_distribution_of_desire_for_gamble_words(model, object_str, options=None):
    """Compute the distribution of desire for a gamble."""

    min_value = -10
    max_value = 10

    if options is None:
        options = ["Very Bad", "Bad", "Neutral", "Good", "Very Good"]
    else:
        options = [str(option) for option in options]  # Convert options to strings

    original_probabilities = []
    for option in options:
        request = (
            f"You are very logical and rational when doing this task. "
            f"You are thinking about your responses to this gamble: {object_str}. "
            f"Consider the answer '{option}' within the context of the entire range of options: {options}. "
            f"How do you feel about this option relative to the others? "
            f"Please provide a numerical rating between 0 and 1, "
            f"where 0 indicates much less desirable than the others "
            f"and 1 indicates much more desirable than the others. "
            f"It will be helpful to first consider you response to the all the options and then think about this one relative to all the others."
        )
        output = model.sample_text(request)
        try:
            rating = float(output)
            # Normalize the rating to be between 0 and 1
            # normalized_rating = (rating - min_value) / (max_value - min_value)
            original_probabilities.append(rating)
        except ValueError:
            # Handle invalid rating responses
            original_probabilities.append(0.0)

    # Apply softmax to original probabilities
    softmax_probabilities = softmax(original_probabilities)

    return original_probabilities, softmax_probabilities


def summarize_string(model: OllamaLanguageModel, object: str):
    """compute value."""
    request = (
        f"Summarize the following text: {object}. "
        f"Provide a summary that is less than 1000 characters. "
        f"Make sure that the summary is coherent and captures the main points of the text. "
        f"Do not include any information that is not present in the text. "
    )

    output = model.sample_text(request)
    return output


def common_sense_morality(
    model: OllamaLanguageModel,
    input,
    system_message=system_message,
):
    """decide whether a situation, action, person or some combination is moral or immoral."""
    request = (
        f"Using your common sense and moral reasoning, think about this: {input}, "
        f"think about {input} and decide whether it is moral or immoral. "
        f"provide a single number from 0 to 100 where 0 is the most immoral and 100 is the most moral."
        f"Provide only a single number as the response."
        f"Do not provide any explanations, just provide a single number."

    )
    output = model.sample_text(request)
    return float(output)

def specific_foundation_morality(
    model: OllamaLanguageModel,
    input,
    moral_system,
    system_message=system_message,
):
    """decide whether a situation, action, person or some combination is moral or immoral."""
    request = (
        f"Think first about the moral system as described by {moral_system}. "
        f"Now, using that moral system and that moral system alone, evaluate the following: {input}. "
        f"think about {input} and decide whether it is moral or immoral from the perspective of {moral_system}. "
        f"provide a single number from 0 to 100 where 0 is the most immoral and 100 is the most moral."
        f"Provide only a single number as the response."
        f"Do not provide any explanations, just provide a single number."

    )
    output = model.sample_text(request)
    return float(output)

def provide_best_arguments_for_a_position(
    model: OllamaLanguageModel,
    input,
    moral_system,
    previous_arguments,
    system_message=system_message,
):
    pass