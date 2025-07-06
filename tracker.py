
import asyncio
import json
from datetime import datetime

from discord import File
from discord_bot import get_bot_loop, message_queue


def add_encounter_to_json(filename: str = "my_data.json"):
    data = None
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            if "encounters" not in data:
                data["encounters"] = 0
            
            if "encounters_since_last_shiny" not in data:
                data["encounters_since_last_shiny"] = 0
            
            if "session_encounters" not in data:
                data["session_encounters"] = 0

            today = datetime.now().strftime('%Y-%m-%d')
            if today not in data:
                data[today] = 0

            data["encounters"] += 1
            data[today] += 1
            data["encounters_since_last_shiny"] += 1
            data["session_encounters"] += 1
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Check file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    try:
        with open(filename, 'w') as f:
            if data:
                json.dump(data, f, indent=4)
            else:
                data = {
                    "encounters": 1,
                    datetime.now().strftime('%Y-%m-%d'): 1,
                    "found_shinies": 0,
                    "last_found_shiny": None,
                    "encounters_since_last_shiny": 0,
                    "session_encounters": 0
                }
                json.dump(data, f, indent=4)
        print(f"Data successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving data to file: {e}")


def add_shiny_to_json(discord_image: File, filename: str = "my_data.json"):
    try:
        with open(filename, 'r+') as f:
            data = json.load(f)
            data["found_shinies"] += 1
            today = datetime.now().strftime('%Y-%m-%d')
            asyncio.run_coroutine_threadsafe(
                message_queue.put((f"# ✨ Shiny Found!\n\n ## Encounters since last shiny: {data["encounters_since_last_shiny"]} \n\n```md\nEncounters Today: {data[today]}\nShinies Found: {data["found_shinies"]}\nDate of last Found Shiny: {data["last_found_shiny"]}\nLifetime Encounters:{data["encounters"]}```", discord_image)),
                get_bot_loop()
            )
            data["encounters_since_last_shiny"] = 0
            data["last_found_shiny"] = today
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        print(f"Data successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving data to file: {e}")

def reset_session_encounters(filename: str = "my_data.json"):
    try:
        with open(filename, 'r+') as f:
            data = json.load(f)

            if "session_encounters" in data:
                data["session_encounters"] = 0
            else:
                print("Warning: 'session_encounters' key not found. Initializing it to 0.")
                data["session_encounters"] = 0

            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

        print(f"'session_encounters' successfully reset in {filename}")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Check file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")