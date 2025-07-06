# ✨ Shiny Hunting Automation Bot

A Python-based automation tool designed to **detect shiny Pokémon encounters** during gameplay using **image recognition and Discord integration**. It captures screen data, analyzes battle sprites using ORB and color matching, and alerts you via Discord when a shiny is found.

---

## 📸 Features

- 🎮 Real-time screen capture and battle detection  
- 🧠 Shiny recognition via ORB matching and color analysis  
- 📈 Tracks encounters, daily stats, and shinies  
- 💬 Sends Discord alerts with screenshots of shinies  
- 📁 Automatically saves battle sprite screenshots  
- 🧵 Multi-threaded for performance and responsiveness

---

## 📁 Project Structure

```
.
├── discord_bot.py         # Discord integration and message handling
├── tracker.py             # Encounter/shiny tracking logic
├── main.py                # Core automation and shiny detection
├── .env                   # Contains your Discord bot token and channel ID
├── my_data.json           # Data file storing encounter stats
├── /Shiny                 # Folder containing reference shiny sprites
├── /screenshots           # Captured screenshots for review
└── README.md              # This file
```

---

## ⚙️ Setup

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Typical dependencies</summary>

```
opencv-python
numpy
discord.py
python-dotenv
scikit-image
scikit-learn
pyautogui
pillow
```
</details>

---

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```
DISCORD_TOKEN=your_discord_bot_token
CHANNEL_ID=your_discord_channel_id
```

---

### 3. Add Shiny Reference Sprites

Place PNG images of shiny Pokémon in the `./Shiny/` folder. These are used for image comparison during battles.

---

### 4. Run the Bot

Start the bot with:

```bash
python main.py
```

The bot will begin:
- Watching your screen for battle triggers
- Capturing battle sprites
- Comparing them to shiny references
- Sending a Discord alert if a shiny is found 🎉

---

## 🧠 How It Works

1. **Battle Detection**  
   A pixel monitor watches the screen for a specific pixel color that indicates a battle has started.

2. **Battle Sprite Capture**  
   When a battle is detected, a portion of the screen is captured to isolate the Pokémon sprite.

3. **Shiny Comparison**  
   The captured image is compared against your shiny sprite references using:
   - **ORB feature matching**
   - **Dominant color clustering** with KMeans
   - **Color distance scoring**

4. **Discord Notifications**  
   If a match is found (based on color similarity and feature matches):
   - The shiny sprite is saved
   - A screenshot and message are sent to your configured Discord channel

5. **Data Logging**  
   All encounters and shiny finds are logged in `my_data.json`:
   ```json
   {
     "encounters": 1450,
     "found_shinies": 3,
     "encounters_since_last_shiny": 78,
     "last_found_shiny": "2025-07-04",
     "2025-07-05": 22
   }
   ```

---

## 📊 Example Discord Output

```
# 🤖 Bot is now online and hunting shinies!

```md
Encounters Today: 45
Date of last Found Shiny: 2025-07-04
Lifetime Encounters: 1523
```
```

When a shiny is found:

```
# ✨ Shiny Found!

## Encounters since last shiny: 93

```md
Encounters Today: 61
Shinies Found: 4
Date of last Found Shiny: 2025-07-05
Lifetime Encounters: 1584
```
```

(Plus a screenshot image!)

---

## 🔧 Notes

- Make sure your emulator/game is in the expected screen region (adjust `region` in `main.py` if needed)
- Discord shutdown/startup messages are sent automatically
- Uses `pyautogui` for key presses, so the game must be the active window

---

## 🧪 Troubleshooting

- **No shiny detected?** Try adding clearer or more accurate reference images to `./Shiny/`.
  - It should be the front facing battle sprite for the current generation
- **Wrong screen coordinates?** Adjust the `region` or pixel positions in `main.py`.
  - It should be set to capture the section of the opposing battle sprite
- **Discord messages not appearing?** Double-check your `.env` and make sure the bot has access to the channel.

---

## 📜 License

***TODO***

---

## 🙏 Acknowledgements

Built with ❤️ using OpenCV, Discord.py, PyAutoGUI, and scikit-learn. Inspired by the hunt for shinies that never ends.
