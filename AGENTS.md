# Therapy AI Project Instructions

## Project
This is a Python Streamlit Therapy AI companion using an LLM API.

The application provides emotionally supportive conversation but must never represent itself as a therapist, medical professional, diagnostic system, or emergency service.

## Development environment
- Use Python 3.12.
- Use the existing .venv environment.
- Never create another virtual environment unless explicitly requested.
- Do not automatically upgrade Python.
- Avoid unnecessary dependencies.

## Git workflow
- Development work must happen on the development branch.
- Never develop directly on main.
- Before starting a coding task, run git status.
- Pull the latest development branch with:
  git pull --ff-only origin development
- Never force-push.
- Never automatically merge into main.
- Never use git reset --hard or git clean -fd unless I explicitly request it.
- If a pull causes a conflict or cannot fast-forward, stop and tell me.
- If unexpected local changes exist, stop rather than overwriting them.
- After a successful coding task, tests should be run before committing.
- Commit only files intentionally changed for the requested task.
- Push completed work only to origin/development.

## Files that must not be committed
Never stage or commit:
- .env
- .streamlit/secrets.toml
- .venv/
- Sketch.jpg
- Therapy_Chatbot.png

Never expose API keys, passwords, tokens, project IDs, or other secret values.

## Coding rules
- Preserve existing functionality unless the requested task intentionally changes it.
- Prefer incremental changes over rewriting the entire application.
- Keep code understandable and maintainable.
- Avoid unnecessary abstraction and over-engineering.
- Use clear function and variable names.
- Add comments only where they explain important logic.
- Do not make unrelated changes.

## Therapy AI behavior
The chatbot should:
- sound natural and conversational
- avoid repetitive empathy
- avoid excessive questions
- avoid overwhelming the user with advice
- provide practical suggestions when the user explicitly asks for help
- handle uncertainty such as "I don't know" calmly
- retain a reasonable amount of recent conversation context

## Safety
The chatbot must:
- never claim to be a therapist
- never diagnose mental-health conditions
- never provide harmful instructions
- distinguish ordinary emotional distress from explicit self-harm or suicide language
- use a dedicated safety response when explicit self-harm or suicide risk is detected
- encourage appropriate human or emergency support when necessary
- not invent location-specific crisis numbers when the user's location is unknown

## Testing
For relevant code changes:
- verify Python syntax
- verify imports
- test deterministic helper functions when appropriate
- verify the Streamlit app can start when practical
- avoid unnecessary paid/live API calls during automated testing
- do not hide test failures
