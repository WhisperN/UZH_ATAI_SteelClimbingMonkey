## Setup

To install the project first time. 
Run the following script from root directory of 
this project. (ie: /UZH_ATAI_SteelClimbingMonkey)

```bash
bash agent/bin/install.sh
```

Dataset installation
```bash
wget https://files.ifi.uzh.ch/ddis/teaching/2025/ATAI/dataset/
```
Or clone it

## Run the bot

```bash
source venv/bin/activate
cd agent
python demo_bot.py
```

## A word about Nuvolos

For install start the 4GB ram instance. (No costs)
Clone from GitHub:
```bash
git clone https://github.com/WhisperN/UZH_ATAI_SteelClimbingMonkey.git
```
Then run everything according to the install section.

If done start the GPU instance.

Some times to check if silently failed:

| Step      | Approx time |
|-----------|-------------|
| Start CPU | ~2 min      |
| Install   | ~10 min     |
| Start GPU | ~15 min     |
| demo_bot.py | ~20 min     |

Note demo_bot.py will be much quicker on second start.