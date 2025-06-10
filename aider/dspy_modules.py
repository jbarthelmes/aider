import dspy
from aider import prompts

class GenerateCommitMessage(dspy.Signature):
    """Generate a concise, one-line Git commit message based on provided diffs and context."""
    diff_content = dspy.InputField(desc="The content of the diffs.")
    context_info = dspy.InputField(desc="Additional context for the commit.")
    language_instruction_info = dspy.InputField(desc="Instruction regarding the language of the commit message.")
    commit_message = dspy.OutputField(desc="A one-line conventional commit message.")

class DspyCommit(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_commit = dspy.Predict(GenerateCommitMessage, prompt_template="""
You are an expert software engineer that generates concise, one-line Git commit messages based on the provided diffs.
Review the provided context and diffs which are about to be committed to a git repo.
Review the diffs carefully.
Generate a one-line commit message for those changes.
The commit message should be structured as follows: <type>: <description>
Use these for <type>: fix, feat, build, chore, ci, docs, style, refactor, perf, test

Ensure the commit message:{{language_instruction_info}}
- Starts with the appropriate prefix.
- Is in the imperative mood (e.g., "add feature" not "added feature" or "adding feature").
- Does not exceed 72 characters.

Reply only with the one-line commit message, without any additional text, explanations, or line breaks.

Context:
{{context_info}}

Diffs:
{{diff_content}}

Commit Message:
""")

    def forward(self, diff_content, context_info, language_instruction_info):
        # language_instruction_info is now expected to be pre-formatted by the caller
        # (e.g., "\n- Is written in German." or an empty string)
        prediction = self.generate_commit(
            diff_content=diff_content,
            context_info=context_info,
            language_instruction_info=language_instruction_info
        )
        return prediction


class AskSignature(dspy.Signature):
    """Answer questions about the codebase or general programming topics."""
    system_prompt_main = dspy.InputField(desc="The main system prompt for answering questions.")
    repo_context = dspy.InputField(desc="Context from repo map, read-only files, and editable files, used for answering code-related questions.", default="")
    chat_history = dspy.InputField(desc="Previous turns of the conversation.", default="")
    user_request = dspy.InputField(desc="The user's question.")
    system_reminder_final = dspy.InputField(desc="A final system reminder for question answering.", default="")
    assistant_response = dspy.OutputField(desc="The assistant's textual answer to the user's question.")


class DspyAskPredictor(dspy.Module):
    def __init__(self, num_outputs=1):
        super().__init__()
        # This prompt template uses the standard structure.
        # The behavior of answering questions is primarily determined by:
        # 1. The content of `system_prompt_main` (from AskPrompts.main_system).
        # 2. The nature of the `user_request` (a question).
        # Since AskPrompts has no specific examples, few-shot behavior would rely on
        # general conversational examples or context provided if a teleprompter is used.
        self.predictor = dspy.Predict(
            AskSignature,
            num_outputs=num_outputs,
            prompt_template="""System: {{system_prompt_main}}

{{repo_context}}

{{chat_history}}

User: {{user_request}}

{{system_reminder_final}}
Assistant:"""
        )

    def forward(self, system_prompt_main, user_request, repo_context="", chat_history="", system_reminder_final=""):
        # Ensure default values are handled if not provided.
        repo_context = repo_context or ""
        chat_history = chat_history or ""
        system_reminder_final = system_reminder_final or ""

        prediction = self.predictor(
            system_prompt_main=system_prompt_main,
            repo_context=repo_context,
            chat_history=chat_history,
            user_request=user_request,
            system_reminder_final=system_reminder_final
        )
        return prediction


class UdiffSignature(dspy.Signature):
    """Generate code changes in unified diff format."""
    system_prompt_main = dspy.InputField(desc="The main system prompt for generating unified diffs.")
    repo_context = dspy.InputField(desc="Context from repo map, read-only files, and editable files.", default="")
    chat_history = dspy.InputField(desc="Previous turns of the conversation.", default="")
    user_request = dspy.InputField(desc="The current user's request for code changes.")
    system_reminder_final = dspy.InputField(desc="A final system reminder for unified diff generation.", default="")
    assistant_response = dspy.OutputField(desc="The assistant's response, containing code changes in unified diff format.")


class DspyUdiffPredictor(dspy.Module):
    def __init__(self, num_outputs=1):
        super().__init__()
        # This prompt template is identical in structure to the EditBlockPredictor's
        # and WholeFilePredictor's template. The difference in behavior (edit block,
        # whole file, or unified diff) is primarily determined by:
        # 1. The content of `system_prompt_main`.
        # 2. The few-shot examples provided to the LM.
        # 3. The user's request itself.
        self.predictor = dspy.Predict(
            UdiffSignature,
            num_outputs=num_outputs,
            prompt_template="""System: {{system_prompt_main}}

{{repo_context}}

{{chat_history}}

User: {{user_request}}

{{system_reminder_final}}
Assistant:"""
        )

    def forward(self, system_prompt_main, user_request, repo_context="", chat_history="", system_reminder_final=""):
        # Ensure default values are handled if not provided.
        repo_context = repo_context or ""
        chat_history = chat_history or ""
        system_reminder_final = system_reminder_final or ""

        prediction = self.predictor(
            system_prompt_main=system_prompt_main,
            repo_context=repo_context,
            chat_history=chat_history,
            user_request=user_request,
            system_reminder_final=system_reminder_final
        )
        return prediction


class WholeFileSignature(dspy.Signature):
    """Generate whole file content based on user request and code context."""
    # Inputs
    system_prompt_main = dspy.InputField(desc="The main part of the system prompt, including core instructions for whole file edits and formatted examples if they are part of the system message.")
    repo_context = dspy.InputField(desc="Context from repo map, read-only files, and editable files, formatted as a block of text or structured user/assistant turns.", default="")
    chat_history = dspy.InputField(desc="Previous turns of the conversation, formatted.", default="")
    user_request = dspy.InputField(desc="The current user's request for code changes.")
    system_reminder_final = dspy.InputField(desc="A final system reminder for whole file edits, appended before the assistant's expected response.", default="")

    # Output
    assistant_response = dspy.OutputField(desc="The assistant's response, containing entire file contents for each modified file.")


class DspyWholeFilePredictor(dspy.Module):
    def __init__(self, num_outputs=1):
        super().__init__()
        # This prompt template is identical in structure to the EditBlockPredictor's template.
        # The difference in behavior (edit block vs whole file) is primarily determined by:
        # 1. The content of `system_prompt_main` (which will be different for whole file edits).
        # 2. The few-shot examples provided to the LM during configuration (if any).
        # 3. The user's request itself, which might explicitly ask for whole file changes.
        self.predictor = dspy.Predict(
            WholeFileSignature,
            num_outputs=num_outputs,
            prompt_template="""System: {{system_prompt_main}}

{{repo_context}}

{{chat_history}}

User: {{user_request}}

{{system_reminder_final}}
Assistant:"""
        )

    def forward(self, system_prompt_main, user_request, repo_context="", chat_history="", system_reminder_final=""):
        # Ensure default values are handled if not provided.
        repo_context = repo_context or ""
        chat_history = chat_history or ""
        system_reminder_final = system_reminder_final or ""

        prediction = self.predictor(
            system_prompt_main=system_prompt_main,
            repo_context=repo_context,
            chat_history=chat_history,
            user_request=user_request,
            system_reminder_final=system_reminder_final
        )
        return prediction


class EditBlockSignature(dspy.Signature):
    """Generate code edits using SEARCH/REPLACE blocks."""
    # Inputs
    system_prompt_main = dspy.InputField(desc="The main part of the system prompt, including core instructions and formatted examples if they are part of the system message.")
    repo_context = dspy.InputField(desc="Context from repo map, read-only files, and editable files, formatted as a block of text or structured user/assistant turns.", default="")
    chat_history = dspy.InputField(desc="Previous turns of the conversation, formatted.", default="")
    user_request = dspy.InputField(desc="The current user's request for code changes.")
    system_reminder_final = dspy.InputField(desc="A final system reminder appended before the assistant's expected response.", default="")

    # Output
    assistant_response = dspy.OutputField(desc="The assistant's response, including explanations and SEARCH/REPLACE blocks.")


class DspyEditBlockPredictor(dspy.Module):
    def __init__(self, num_outputs=1):
        super().__init__()
        # The prompt template is designed to mimic the structure of a typical chat conversation
        # where different types of information (system instructions, context, history, current request)
        # are presented in sequence.
        # The {{field_name}} placeholders will be filled by DSPy from the input fields
        # of the EditBlockSignature.
        # The 'Assistant:' at the end cues the LM to generate the assistant's response.
        self.predictor = dspy.Predict(
            EditBlockSignature,
            num_outputs=num_outputs,
            prompt_template="""System: {{system_prompt_main}}

{{repo_context}}

{{chat_history}}

User: {{user_request}}

{{system_reminder_final}}
Assistant:"""
        )

    def forward(self, system_prompt_main, user_request, repo_context="", chat_history="", system_reminder_final=""):
        # Ensure default values are handled if not provided, though dspy.InputField defaults should also work.
        repo_context = repo_context or ""
        chat_history = chat_history or ""
        system_reminder_final = system_reminder_final or ""

        prediction = self.predictor(
            system_prompt_main=system_prompt_main,
            repo_context=repo_context,
            chat_history=chat_history,
            user_request=user_request,
            system_reminder_final=system_reminder_final
        )
        return prediction
