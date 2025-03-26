import sys
import warnings
warnings.filterwarnings("ignore")

ANSI_RESET="\033[0m"        # Reset all formatting
ANSI_BOLD="\033[1m"         # Bold text
ANSI_YELLOW="\033[33m"      # Yellow text

sys.stdout.write(ANSI_BOLD + ANSI_YELLOW)
# print("Training HuggingFace's Whisper Model:")
import pyfiglet
distil_whisper_art = pyfiglet.figlet_format("whisper-training",  font="slant", justify="center", width=100)
# print(distil_whisper_art)

sys.stdout.write(ANSI_RESET)

from src import App

from src.utils.logger import logger, config

def main():
    logger.info("Starting the application...")
    app = App()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        logger.error("Keyboard Interrupted...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Exited due to {e}")
        sys.exit(0)