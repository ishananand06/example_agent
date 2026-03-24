import os
import requests
import json
import re

try:
    from dotenv import load_dotenv
except ImportError as e:
    raise RuntimeError(
        "Missing dependency 'python-dotenv'. Install it with: pip install python-dotenv"
    ) from e

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


REQUEST_TIMEOUT_SECONDS = 10

# ==========================================
# 1. TOOLS (The Hands)
# Specific functions that do a specific task.
# ==========================================

def get_current_location():
    """Returns the current latitude and longitude based on IP."""
    try:
        # Free API, no key required
        response = requests.get("http://ip-api.com/json/", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            return f"Location lookup failed: {data.get('message', 'unknown error')}"
        return f"Latitude: {data['lat']}, Longitude: {data['lon']}, City: {data['city']}"
    except Exception as e:
        return f"Location error: {e}"

def get_weather(latitude, longitude):
    """Returns current weather for a given lat/lon."""
    try:
        if latitude is None or longitude is None:
            return "Weather error: latitude and longitude are required."

        # Free API, no key required
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        current = data.get("current")
        if not current:
            return "Weather error: missing 'current' data in API response."

        temp = current.get("temperature_2m")
        wind = current.get("wind_speed_10m")
        if temp is None or wind is None:
            return "Weather error: incomplete weather fields in API response."
        return f"Temperature: {temp}°C, Wind Speed: {wind} km/h"
    except Exception as e:
        return f"Weather error: {e}"

# ==========================================
# 2. THE BRAIN (LLM Call)
# ==========================================

def call_llm(memory):
    """
    Sends the memory state to the Gemini API. 
    Requires: pip install google-genai
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency 'google-genai'. Install it with: pip install google-genai"
        ) from e

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    
    # Initialize the official Gemini client
    client = genai.Client(api_key=api_key)
    
    # 1. Extract the system prompt (Gemini handles this in the config)
    sys_prompt = next((m["content"] for m in memory if m["role"] == "system"), None)
    
    # 2. Format the memory history for Gemini
    gemini_history = []
    for msg in memory:
        if msg["role"] == "system":
            continue # Skip, we already extracted it
            
        # Map standard 'assistant' role to Gemini's 'model' role
        role = "model" if msg["role"] == "assistant" else msg["role"]
        
        gemini_history.append(
            {"role": role, "parts": [{"text": msg["content"]}]}
        )

    # 3. Call the Gemini model
    response = client.models.generate_content(
        model='gemini-2.5-flash', # Highly recommended for fast agent loops
        contents=gemini_history,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=0.0 # Keep at 0 for strict, predictable reasoning
        )
    )
    
    return response.text


def parse_action_and_args(response_text):
    """Extracts ACT and ARGS from model output in a tolerant way."""
    act_match = re.search(r"^ACT:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", response_text, re.MULTILINE)
    action = act_match.group(1).strip() if act_match else None

    args = {}
    args_header = re.search(r"^ARGS:\s*", response_text, re.MULTILINE)
    if args_header:
        after_args = response_text[args_header.end():].lstrip()
        if after_args.startswith("{"):
            decoder = json.JSONDecoder()
            try:
                parsed_args, _ = decoder.raw_decode(after_args)
                if isinstance(parsed_args, dict):
                    args = parsed_args
            except json.JSONDecodeError:
                pass

    return action, args

# ==========================================
# 3. PLANNER (The System Prompt)
# Instructs the LLM how to reason and output data.
# ==========================================

system_prompt = """
You are an autonomous agent capable of reasoning and using tools. 
You run in a loop of REASON, ACT, and OBSERVE.

Tools available:
1. get_current_location: Returns your current latitude, longitude, and city. No arguments needed.
2. get_weather: Returns weather for a given location. Arguments: latitude (float), longitude (float).

Use the following strict format:

Question: the input question you must answer
REASON: think about what to do next based on the goal
ACT: the action to take (must be one of: get_current_location, get_weather)
ARGS: the arguments for the action in valid JSON format (e.g. {"latitude": 28.61, "longitude": 77.20})
OBSERVE: the result of the action (I will provide this to you)

If you have the final answer, output:
FINAL ANSWER: your final answer to the user.
"""

# ==========================================
# 4. AGENT LOOP & MEMORY (Repeat, Reason, Act, Observe)
# ==========================================

def run_agent(user_query):
    # Initialize Memory (The State)
    memory = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {user_query}"}
    ]
    
    max_loops = 5 # Safety check so the agent doesn't run forever
    
    # REPEAT (Loop until goal is reached)
    for loop in range(max_loops):
        print(f"\n--- 🔄 Agent Loop {loop + 1} ---")
        
        # 🧠 REASON (Think about what to do)
        print("🧠 Thinking...")
        try:
            response_text = call_llm(memory)
        except Exception as e:
            print(f"LLM call failed: {e}")
            memory.append({"role": "assistant", "content": f"FINAL ANSWER: Agent failed with error: {e}"})
            break

        print(f"Agent Output:\n{response_text}\n")
        
        # Add the LLM's thought process to memory
        memory.append({"role": "assistant", "content": response_text})
        
        # Check if the agent reached the goal
        if "FINAL ANSWER:" in response_text:
            print("\n✅ Goal Achieved!")
            break
            
        action, args = parse_action_and_args(response_text)
        if not action:
            observation = "Error: Missing ACT in model output."
            print(f"👀 Observation: {observation}")
            memory.append({"role": "user", "content": f"OBSERVE: {observation}"})
            continue

        # 🔧 ACT (Use the tool)
        print(f"🔧 Executing Tool: {action} with args {args}")
        if action == "get_current_location":
            observation = get_current_location()
        elif action == "get_weather":
            observation = get_weather(args.get("latitude"), args.get("longitude"))
        else:
            observation = "Error: Tool not found."

        # 👀 OBSERVE (See what happened)
        print(f"👀 Observation: {observation}")

        # Feed the observation back into the LLM's memory
        memory.append({"role": "user", "content": f"OBSERVE: {observation}"})
            
    return memory

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Test it with a query that requires multi-step reasoning
    run_agent("Where am I currently located, and what is the exact weather like here in New Delhi right now compared to my IP location?")