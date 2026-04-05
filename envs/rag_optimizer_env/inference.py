import os
import json
from openai import OpenAI

from client import RagOptimizerEnvClient
from models import RagOptimizerAction

# Load environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

MAX_STEPS = 15

SYSTEM_PROMPT = """You are an automated Data Engineer managing an AI Knowledge Base.
Your goal is to optimize the messy chunks of text in the database so that a TF-IDF Search Algorithm can find answers easily.
You must resolve contradictions, categorize documents, and splinter monolithic text blobs into smaller chunks.

You have the following actions:
- {"action_type": "read_document", "doc_id": "..."}
- {"action_type": "update_document", "doc_id": "...", "text": "..."}
- {"action_type": "delete_document", "doc_id": "..."}
- {"action_type": "add_metadata", "doc_id": "...", "metadata_key": "...", "metadata_value": "..."}
- {"action_type": "submit"}

You must return ONLY a raw JSON object detailing the action you want to take!"""

def main():
    # Setup OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Connect to the local OpenEnv server using .sync() since EnvClient is async by default
    print("Connecting to backend OpenEnv Server at http://localhost:8000...")
    with RagOptimizerEnvClient(base_url="http://localhost:8000").sync() as env:
        
        result = env.reset()
        observation = result.observation
        
        print("\n--- ENVIRONMENT RESET ---")
        print(f"Server Message: {observation.message}")
        print(f"Initial KB Stats: {len(observation.current_docs)} files loaded.\n")
        
        history = []
        
        for step in range(1, MAX_STEPS + 1):
            
            # Format the observation for the LLM
            obs_dict = {
                "server_feedback": observation.message,
                "current_knowledge_base": observation.current_docs
            }
            user_content = json.dumps(obs_dict, indent=2)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            try:
                # Call OpenAI SDK for inference exactly like the Hackathon example
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=1000
                )
                response_text = completion.choices[0].message.content or ""
                action_data = json.loads(response_text)
                action = RagOptimizerAction(**action_data)
                
            except Exception as exc: 
                print(f"Model request failed ({exc}). Using fallback action.")
                # Fallback directly to end episode so we don't crash the simulation
                action = RagOptimizerAction(action_type="submit")

            print(f"Step {step}: Agent called -> {action.action_type} on doc_id: {action.doc_id}")

            # Send action to Environment
            result = env.step(action)
            observation = result.observation
            reward = result.reward

            print(f"  Reward Status: {reward:+.2f} | Done: {result.done} | Server: {observation.message}")

            if result.done:
                print("\n=== EPISODE COMPLETE ===")
                print(f"Final Hackathon Score: {reward * 100:.1f}%")
                break

        else:
            print(f"Reached max steps ({MAX_STEPS}). Exiting.")

if __name__ == "__main__":
    main()
