"""
SpaceText — Taxonomy-Aware Aerospace Summarizer (Sarvam-1 Version)
================================================
Interactive CLI for generating multi-document aerospace research summaries.
Replaces the old test_rag.py Q&A interface.

Uses GarudDB's hyperbolic clustering to retrieve an entire topic cluster,
then synthesizes it with the Sarvam-1 (2B) model.

Usage:
    python summarize_sarvam.py
    python summarize_sarvam.py --topic "DRDO inertial navigation systems"
    python summarize_sarvam.py --topic "Chandrayaan lunar missions" --source isro
    python summarize_sarvam.py --topic "Solid rocket propulsion" --top-k 25
    python summarize_sarvam.py --qa "What is a Kalman filter?"
"""

import argparse
import sys
import logging

# Suppress noisy library output
logging.basicConfig(level=logging.WARNING)
logging.getLogger("spacetext.rag").setLevel(logging.INFO)

from spacetext.rag.agent import SpaceTextAgent

# ── ANSI colour codes for terminal output ─────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
DIM    = "\033[2m"
LINE   = f"{DIM}{'─' * 72}{RESET}"


def print_banner():
    print(f"""
{CYAN}{BOLD}
  ███████╗██████╗  █████╗  ██████╗███████╗████████╗███████╗██╗  ██╗████████╗
  ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
  ███████╗██████╔╝███████║██║     █████╗     ██║   █████╗   ╚███╔╝    ██║   
  ╚════██║██╔═══╝ ██╔══██║██║     ██╔══╝     ██║   ██╔══╝   ██╔██╗    ██║   
  ███████║██║     ██║  ██║╚██████╗███████╗   ██║   ███████╗██╔╝ ██╗   ██║   
  ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   
{RESET}
{DIM}  Taxonomy-Aware Aerospace Document Summarizer | Powered by Sarvam-1 (2B) + GarudDB{RESET}
""")


def print_cluster_info(cluster: list, topic: str):
    """Prints a table showing which documents were retrieved."""
    if not cluster:
        print(f"{YELLOW}  ⚠  No documents found for topic: '{topic}'{RESET}\n")
        return

    source_counts = {}
    for src, _ in cluster:
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\n{CYAN}{BOLD}  Retrieved Document Cluster{RESET} — {len(cluster)} documents\n")
    for src, count in sorted(source_counts.items()):
        bar = "█" * count
        print(f"    {YELLOW}{src.upper():12s}{RESET}  {bar} {count}")
    print()


def print_structured_summary(summary: str):
    """Pretty-prints the structured summary with colour coding."""
    print(f"\n{LINE}")
    print(f"\n{CYAN}{BOLD}  📋 STRUCTURED SUMMARY{RESET}\n")
    print(f"{LINE}\n")

    for line in summary.split("\n"):
        if line.startswith("## "):
            print(f"\n{BLUE}{BOLD}{line}{RESET}")
        elif line.startswith("**") and line.endswith("**"):
            print(f"{GREEN}{BOLD}{line}{RESET}")
        elif line.startswith("**"):
            # Bold key: value pairs
            parts = line.split("**: ", 1)
            if len(parts) == 2:
                print(f"{GREEN}{BOLD}{parts[0]}**{RESET}: {parts[1]}")
            else:
                print(line)
        elif line.startswith("- "):
            print(f"  {YELLOW}•{RESET} {line[2:]}")
        elif line.strip():
            print(f"  {line}")
        else:
            print()


def print_narrative_summary(narrative: str):
    """Pretty-prints the narrative synthesis paragraph."""
    if not narrative or len(narrative) < 30:
        return

    print(f"\n{LINE}")
    print(f"\n{MAGENTA}{BOLD}  📖 NARRATIVE SYNTHESIS{RESET}\n")
    print(f"{LINE}\n")

    # Word-wrap at 72 chars
    words = narrative.split()
    line = "  "
    for word in words:
        if len(line) + len(word) + 1 > 74:
            print(line)
            line = "  " + word
        else:
            line += (" " if line.strip() else "") + word
    if line.strip():
        print(line)
    print()


def run_summarization(agent: SpaceTextAgent, topic: str, source_filter: str = None):
    """Runs the full summarization pipeline and prints results."""
    print(f"\n{BOLD}  Topic:{RESET} {topic}")
    if source_filter:
        print(f"  {BOLD}Filter:{RESET} {source_filter.upper()} only")
    print(f"\n  {DIM}Retrieving documents from GarudDB...{RESET}")

    result = agent.summarize(topic=topic, source_filter=source_filter)

    print_cluster_info(result["cluster"], topic)

    if not result["cluster"]:
        print(f"{YELLOW}  No documents were retrieved. Try a broader topic.{RESET}\n")
        return

    print(f"  {DIM}Generating synthesis with Sarvam-1 (2B)...{RESET}")

    print_structured_summary(result["structured_summary"])
    print_narrative_summary(result["narrative_summary"])
    print(f"\n{LINE}\n")


def run_qa(agent: SpaceTextAgent, question: str):
    """Runs Q&A mode and prints results."""
    print(f"\n{BOLD}  Question:{RESET} {question}")
    print(f"\n  {DIM}Retrieving context from GarudDB...{RESET}")

    result = agent.ask(question)

    print(f"\n{LINE}")
    print(f"\n{CYAN}{BOLD}  💬 ANSWER{RESET}\n")
    print(f"{LINE}\n")

    print(f"  {BOLD}Retrieved Documents:{RESET} {len(result['documents'])}")
    for i, doc in enumerate(result["documents"], 1):
        first_line = doc.split("\n")[0]
        print(f"  {DIM}  [{i}] {first_line}{RESET}")

    print()
    for line in result["answer"].split("\n"):
        if line.strip():
            print(f"  {line}")
        else:
            print()

    print(f"\n{LINE}\n")


def interactive_loop(agent: SpaceTextAgent):
    """Runs an interactive REPL for summarization and Q&A."""
    print(f"\n{DIM}  Commands:{RESET}")
    print(f"  {YELLOW}summarize <topic>{RESET}        — Summarize a topic cluster")
    print(f"  {YELLOW}ask <question>{RESET}            — Ask a direct question")
    print(f"  {YELLOW}source <src> <topic>{RESET}      — Summarize filtered by source (isro/drdo/nasa/esa/arxiv)")
    print(f"  {YELLOW}quit{RESET}                      — Exit\n")

    while True:
        try:
            raw = input(f"{CYAN}SpaceText›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}\n")
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in ("quit", "exit", "q"):
            print(f"\n{DIM}Goodbye.{RESET}\n")
            break

        elif lower.startswith("source "):
            parts = raw.split(" ", 2)
            if len(parts) < 3:
                print(f"  {YELLOW}Usage: source <src> <topic>{RESET}\n")
                continue
            _, src, topic = parts
            run_summarization(agent, topic, source_filter=src.lower())

        elif lower.startswith("ask "):
            question = raw[4:].strip()
            if question:
                run_qa(agent, question)
            else:
                print(f"  {YELLOW}Please provide a question.{RESET}\n")

        elif lower.startswith("summarize "):
            topic = raw[10:].strip()
            if topic:
                run_summarization(agent, topic)
            else:
                print(f"  {YELLOW}Please provide a topic.{RESET}\n")

        else:
            # Default: treat as summarization topic
            run_summarization(agent, raw)


def main():
    parser = argparse.ArgumentParser(
        description="SpaceText — Taxonomy-Aware Aerospace Document Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarize_sarvam.py
  python summarize_sarvam.py --topic "DRDO inertial navigation systems"
  python summarize_sarvam.py --topic "Chandrayaan lunar missions" --source isro
  python summarize_sarvam.py --qa "How does a Hall thruster work?"
        """,
    )
    parser.add_argument("--topic",  help="Topic to summarize (skips interactive mode)")
    parser.add_argument("--source", help="Filter by source: isro, drdo, nasa, esa, arxiv")
    parser.add_argument("--qa",     help="Ask a direct question (Q&A mode)")
    parser.add_argument(
        "--model",
        default="./models/sarvam-1-instruct-q4_k_m.gguf",
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--db",
        default="./aerospace_taxonomy.adk",
        help="Path to the GarudDB index",
    )
    args = parser.parse_args()

    print_banner()

    print(f"  {DIM}Loading SpaceText Agent...{RESET}")
    try:
        agent = SpaceTextAgent(model_path=args.model, db_path=args.db)
        print(f"  {GREEN}✓ Agent ready.{RESET}\n")
    except Exception as e:
        print(f"  {YELLOW}✗ Failed to initialize agent: {e}{RESET}\n")
        sys.exit(1)

    if args.qa:
        run_qa(agent, args.qa)
    elif args.topic:
        run_summarization(agent, args.topic, source_filter=args.source)
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()
