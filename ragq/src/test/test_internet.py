import argparse
import asyncio

from src.utils.internet import search_on_baike


parser = argparse.ArgumentParser(description="Search on Baike and save the content.")
parser.add_argument("--query", help="The keyword to search for.")
parser.add_argument("--output", default='.', help="Output directory (default: current directory).")
parser.add_argument("--filename", help="Optional, name of the output file (default: <query>.txt)")

args = parser.parse_args()

asyncio.run(search_on_baike(args.query, args.output, args.filename))