#!/usr/bin/env python3
"""
Website Summarizer using Ollama (Local LLM)
Challenge solution for Week 1 Day 2

This script upgrades the Day 1 website summarizer to use Ollama
instead of OpenAI, allowing you to run everything locally for free.

Usage:
    uv run week1/claude_solution.py

Or with a specific URL:
    uv run week1/claude_solution.py https://example.com
"""

import sys
from openai import OpenAI
from scraper import fetch_website_contents


# Configuration for Ollama
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"  # Use "llama3.2:1b" for slower machines

# System prompt defines the assistant's behavior
SYSTEM_PROMPT = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

# User prompt prefix
USER_PROMPT_PREFIX = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


def create_ollama_client():
    """
    Create an OpenAI client configured to use Ollama's local endpoint.

    Returns:
        OpenAI: Client configured for Ollama
    """
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key='ollama'  # Ollama doesn't need a real API key
    )


def create_messages(website_content):
    """
    Create the message structure expected by the Chat Completions API.

    Args:
        website_content (str): The scraped website content

    Returns:
        list: Messages in the format expected by the API
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_PREFIX + website_content}
    ]


def summarize_website(url, client):
    """
    Fetch a website and generate a summary using Ollama.

    Args:
        url (str): The URL of the website to summarize
        client (OpenAI): The Ollama client

    Returns:
        str: The generated summary in markdown format
    """
    print(f"Fetching website content from: {url}")
    website_content = fetch_website_contents(url)

    print(f"Generating summary using {MODEL}...")
    messages = create_messages(website_content)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return response.choices[0].message.content


def main():
    """
    Main function to run the website summarizer.
    """
    # Default URLs to try
    default_urls = [
        "https://edwarddonner.com",
        "https://anthropic.com",
        "https://cnn.com"
    ]

    # Check if a URL was provided as command line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        urls_to_summarize = [url]
    else:
        print("No URL provided. Summarizing default websites...\n")
        urls_to_summarize = default_urls

    # Create Ollama client
    try:
        client = create_ollama_client()
        print(f"✓ Connected to Ollama at {OLLAMA_BASE_URL}")
        print(f"✓ Using model: {MODEL}\n")
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Visit http://localhost:11434 to check if Ollama is running")
        print("  2. If not, run 'ollama serve' in a terminal")
        print(f"  3. Pull the model with 'ollama pull {MODEL}'")
        sys.exit(1)

    # Summarize each URL
    for url in urls_to_summarize:
        print("=" * 70)
        try:
            summary = summarize_website(url, client)
            print(f"\n📄 Summary of {url}:\n")
            print(summary)
            print()
        except Exception as e:
            print(f"✗ Error summarizing {url}: {e}\n")

    print("=" * 70)
    print("\n✓ Done!")
    print("\nBenefits of using Ollama:")
    print("  • No API charges - completely free")
    print("  • Data stays on your machine - privacy")
    print("  • Works offline")
    print("\nTrade-offs:")
    print("  • Less powerful than frontier models like GPT-4")
    print("  • Slower on consumer hardware")


if __name__ == "__main__":
    main()
