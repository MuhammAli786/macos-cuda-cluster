"""Long-context test prompt: synthetic log entries with one hidden fact."""
import random, sys
from transformers import AutoTokenizer  # noqa: F401  (only to fail fast if missing)
target, out = int(sys.argv[1]), sys.argv[2]
random.seed(7)
towns = ["Halden", "Tarvik", "Mirabel", "Oskar", "Lune", "Brecon", "Valka", "Sorrel", "Amberley", "Quill"]
ports = ["Rotterdam", "Gdansk", "Antwerp", "Lisbon", "Genoa", "Oslo", "Hull", "Bilbao"]
goods = ["copper wire", "rye flour", "glass jars", "wool bales", "tin plates", "cedar planks", "salt sacks"]
lines, n = [], 0
while n < target * 4:  # ~4 chars per token
    i = len(lines) + 1
    l = (f"Log {i:06d}: the depot in {random.choice(towns)} sent {random.randint(2, 99)} crates of "
         f"{random.choice(goods)} to {random.choice(ports)} on day {random.randint(1, 365)}, "
         f"inspector {random.choice('ABCDEFGHJK')}{random.randint(10, 99)} signed it.")
    lines.append(l); n += len(l) + 1
lines.insert(int(len(lines) * 0.6), "Note: the access code for the blue vault is 4826-TANGO. Keep it safe.")
text = "\n".join(lines) + "\n\nQuestion: what is the access code for the blue vault? Answer with the code only."
open(out, "w").write(text)
