import os
import logging
import argparse

from trainer.model import LLMStoryTeller

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def format(txt: str) -> str:
    t = txt.replace('. "', '.\n\n"')
    t = t.replace('" "', '"\n\n"')

    return t


# python -m trainer.generate --model 'mickume/harry_potter_small_model' --prompt '"Well, you’re not a doctor,"'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate text from a finetuned model.")

    # required
    parser.add_argument("--prompt", type=str,
                        help="The prompt to generate from.")
    parser.add_argument("--model", type=str,
                        help="The name of the finetuned model.")

    # optional
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--temp', type=float, default=0.55)
    parser.add_argument('--overlap', type=int, default=6)
    parser.add_argument('--max-sequence', type=int, default=25)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.92)
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()

    story_teller = LLMStoryTeller(args.model)

    print("")
    print(f"PROMPT: {args.prompt}\n")

    text = story_teller.generate(args.prompt, args.max_tokens, args.overlap,
                                 args.temp, args.max_sequence, args.top_k, args.top_p, args.verbose)

    print(format(text))
