import logging
import sys

# Create a logger
logging.basicConfig(filename="log.txt", filemode="w", level=logging.INFO)
logger = logging.getLogger("LOG")
logger.setLevel(logging.INFO)  # Set the minimum logging level
# Create a console handler and set the level to debug
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Add the formatter to the console handler
console_handler.setFormatter(formatter)
# Add the console handler to the logger
logger.addHandler(console_handler)
