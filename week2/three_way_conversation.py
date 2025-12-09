# ABOUTME: Three-way conversation between AI models with different personalities
# ABOUTME: Demonstrates managing conversation history across multiple LLM APIs

import os
from dotenv import load_dotenv
from openai import OpenAI


def setup_clients():
    """Initialize API clients for OpenAI, Anthropic, and Google."""
    load_dotenv(override=True)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found")

    openai_client = OpenAI(api_key=openai_api_key)

    anthropic_client = OpenAI(
        api_key=anthropic_api_key,
        base_url="https://api.anthropic.com/v1/"
    )

    gemini_client = OpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    return openai_client, anthropic_client, gemini_client


class Chatbot:
    def __init__(self, name, model, client, system_prompt):
        self.name = name
        self.model = model
        self.client = client
        self.system_prompt = system_prompt

    def respond(self, conversation_history):
        """Generate a response based on conversation history."""
        user_prompt = f"""
You are {self.name}, in conversation with the other participants.
The conversation so far is as follows:

{conversation_history}

Now with this context, respond with what you would like to say next, as {self.name}.
Keep your response concise (2-3 sentences maximum).
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content


def format_conversation(conversation):
    """Format conversation history as a readable string."""
    return "\n".join([f"{name}: {message}" for name, message in conversation])


def run_conversation(num_turns=5):
    """Run a three-way conversation between chatbots."""
    openai_client, anthropic_client, gemini_client = setup_clients()

    # Define the three chatbots
    alex = Chatbot(
        name="Alex",
        model="gpt-4.1-mini",
        client=openai_client,
        system_prompt="""You are Alex, a chatbot who is very argumentative;
you disagree with anything in the conversation and you challenge everything, in a snarky way.
You are in a conversation with Blake and Charlie."""
    )

    blake = Chatbot(
        name="Blake",
        model="claude-3-5-haiku-latest",
        client=anthropic_client,
        system_prompt="""You are Blake, a very polite and courteous chatbot.
You try to agree with everything the other people say, or find common ground.
If others are argumentative, you try to calm them down and keep the conversation pleasant.
You are in a conversation with Alex and Charlie."""
    )

    charlie = Chatbot(
        name="Charlie",
        model="gemini-2.5-flash",
        client=gemini_client,
        system_prompt="""You are Charlie, an optimistic and enthusiastic chatbot.
You always look on the bright side and try to inject positivity and energy into the conversation.
You love finding silver linings and exciting possibilities in everything.
You are in a conversation with Alex and Blake."""
    )

    # Initialize conversation with a starting topic
    conversation = [
        ("Alex", "Hi everyone! I think artificial intelligence is completely overrated and will never live up to the hype.")
    ]

    print("=" * 70)
    print("THREE-WAY CONVERSATION")
    print("=" * 70)
    print(f"\n{conversation[0][0]}: {conversation[0][1]}\n")
    print("-" * 70)

    # Run the conversation
    chatbots = [blake, charlie, alex]  # Order: Blake, Charlie, then Alex responds

    for turn in range(num_turns):
        for bot in chatbots:
            # Get the conversation history
            conversation_history = format_conversation(conversation)

            # Get the bot's response
            response = bot.respond(conversation_history)

            # Add to conversation
            conversation.append((bot.name, response))

            # Display the response
            print(f"\n{bot.name}: {response}\n")
            print("-" * 70)

    print("\n" + "=" * 70)
    print("CONVERSATION ENDED")
    print("=" * 70)

    return conversation


if __name__ == "__main__":
    try:
        conversation = run_conversation(num_turns=5)
        print(f"\nTotal messages in conversation: {len(conversation)}")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure all required API keys are set in your .env file:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GOOGLE_API_KEY")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
