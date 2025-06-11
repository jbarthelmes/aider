import argparse
import os
import dspy
from dspy.models.openai import OpenAI as DSPyOpenAI
from aider import models, prompts
from aider.dspy_modules import DspyChatSummarizer
from aider.dump import dump  # noqa: F401


class ChatSummary:
    def __init__(self, models=None, max_tokens=1024):
        if not models:
            raise ValueError("At least one model must be provided")
        self.models = models if isinstance(models, list) else [models]
        self.max_tokens = max_tokens
        self.token_count = self.models[0].token_count

    def too_big(self, messages):
        sized = self.tokenize(messages)
        total = sum(tokens for tokens, _msg in sized)
        return total > self.max_tokens

    def tokenize(self, messages):
        sized = []
        for msg in messages:
            tokens = self.token_count(msg)
            sized.append((tokens, msg))
        return sized

    def summarize(self, messages, depth=0):
        messages = self.summarize_real(messages)
        if messages and messages[-1]["role"] != "assistant":
            messages.append(dict(role="assistant", content="Ok."))
        return messages

    def summarize_real(self, messages, depth=0):
        if not self.models:
            raise ValueError("No models available for summarization")

        sized = self.tokenize(messages)
        total = sum(tokens for tokens, _msg in sized)
        if total <= self.max_tokens and depth == 0:
            return messages

        min_split = 4
        if len(messages) <= min_split or depth > 3:
            return self.summarize_all(messages)

        tail_tokens = 0
        split_index = len(messages)
        half_max_tokens = self.max_tokens // 2

        # Iterate over the messages in reverse order
        for i in range(len(sized) - 1, -1, -1):
            tokens, _msg = sized[i]
            if tail_tokens + tokens < half_max_tokens:
                tail_tokens += tokens
                split_index = i
            else:
                break

        # Ensure the head ends with an assistant message
        while messages[split_index - 1]["role"] != "assistant" and split_index > 1:
            split_index -= 1

        if split_index <= min_split:
            return self.summarize_all(messages)

        head = messages[:split_index]
        tail = messages[split_index:]

        sized = sized[:split_index]
        head.reverse()
        sized.reverse()
        keep = []
        total = 0

        # These sometimes come set with value = None
        model_max_input_tokens = self.models[0].info.get("max_input_tokens") or 4096
        model_max_input_tokens -= 512

        for i in range(split_index):
            total += sized[i][0]
            if total > model_max_input_tokens:
                break
            keep.append(head[i])

        keep.reverse()

        summary = self.summarize_all(keep)

        tail_tokens = sum(tokens for tokens, msg in sized[split_index:])
        summary_tokens = self.token_count(summary)

        result = summary + tail
        if summary_tokens + tail_tokens < self.max_tokens:
            return result

        return self.summarize_real(result, depth + 1)

    def summarize_all(self, messages):
        content = ""
        for msg in messages:
            role = msg["role"].upper()
            if role not in ("USER", "ASSISTANT"):
                continue
            content += f"# {role}\n"
            content += msg["content"]
            if not content.endswith("\n"):
                content += "\n"

        dspy_summarizer = DspyChatSummarizer()
        configured_lm = None

        # Configure DSPy LM (using the first available model from self.models)
        # This logic attempts to configure dspy.settings.lm only if it's not already set globally.
        # If a global LM is already configured, this summarizer will use it.
        if not dspy.settings.lm and self.models:
            try:
                primary_model = self.models[0]
                active_model_name = primary_model.name
                if active_model_name.startswith("openai/"): # Or other prefixes if they exist
                    active_model_name = active_model_name.split("/", 1)[1]

                api_key = getattr(primary_model, 'api_key', os.getenv("OPENAI_API_KEY"))
                base_url = getattr(primary_model, 'api_base', None)
                max_output_toks = getattr(primary_model.info, 'max_output_tokens', 1024)
                http_client = getattr(primary_model, 'http_client', None)

                configured_lm = DSPyOpenAI(
                    model=active_model_name,
                    api_key=api_key,
                    api_base=base_url,
                    max_tokens=max_output_toks,
                    temperature=0.0, # Summaries should be deterministic
                    http_client=http_client
                )
                dspy.settings.configure(lm=configured_lm)
            except Exception as e:
                print(f"DSPy LM configuration for summarizer failed: {e}")
                # If config fails, we might still proceed if a global LM was already set.
                # If not, the next check will catch it.

        if not dspy.settings.lm: # Check if LM is configured (either by above block or globally)
             raise ValueError("DSPy LM not configured for summarizer and no models available/suitable to configure it.")

        try:
            prediction = dspy_summarizer(conversation_history=content)
            summary = prediction.summary
            if summary is not None: # Ensure summary is not None before prefixing
                summary = prompts.summary_prefix + summary
                return [dict(role="user", content=summary)]
        except Exception as e:
            # Log the error from DSPy summarization
            print(f"DSPy summarization failed: {str(e)}")
            # Fallback to original model loop if desired, or just raise error.
            # For this refactoring, we'll stick to raising an error if the DSPy path fails,
            # similar to the original behavior of raising if all models failed.
            # If self.models has multiple, and want to try them with DSPy, this loop needs to be here.
            # However, DSPy's typical pattern is one configured LM.
            # For now, if the primary configured DSPy LM fails, we raise.
            pass # Will fall through to the ValueError below if summary is None

        raise ValueError("DSPy summarizer unexpectedly failed or returned no summary.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Markdown file to parse")
    args = parser.parse_args()

    model_names = ["gpt-3.5-turbo", "gpt-4"]  # Add more model names as needed
    model_list = [models.Model(name) for name in model_names]
    summarizer = ChatSummary(model_list)

    with open(args.filename, "r") as f:
        text = f.read()

    summary = summarizer.summarize_chat_history_markdown(text)
    dump(summary)


if __name__ == "__main__":
    main()
