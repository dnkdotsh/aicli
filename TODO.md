# AICLI TODO

Random mix of bugs to fix, features I want, and crazy ideas I might try.

## Stuff That's Actually Broken
- Model validation when switching engines
  - Prevent those annoying 404s when switching from OpenAI to Gemini with wrong model names
- Streaming response edge cases
  - Sometimes the API sends malformed chunks that break the parser

## Features I Actually Want to Build
- Interactive image prompt helper
  - Chat with AI to craft better prompts before spending $0.20 on garbage images
- Claude support
  - Add Anthropic's API so all the AIs can argue with each other
- Export conversations to markdown/PDF
  - Save important chats without copy-pasting everything
- Better setup wizard
  - Current first-run is confusing for non-technical users
- Batch mode for processing multiple prompts
  - Feed it a file of prompts, get a file of responses

## UI Stuff That Would Be Nice
- Progress bars
  - So I know it's working and not frozen
- Better error messages
  - Actually explain what went wrong and how to fix it
- File preview before attaching
  - Avoid accidentally attaching sensitive stuff
- Tab completion for slash commands
  - Because typing `/personas` gets old

## The Big Refactor (Client-Server Split)
- Turn this into a proper server + lightweight clients
  - Business logic on server, UI just sends HTTP requests
- Web app that talks to my server
  - React/Vue frontend, way easier than desktop apps
- Mobile apps
  - Same server API, different UI for phones
- Let people use their own keys OR pay for mine
  - Freemium model: BYOK or premium subscription

## GUI Dreams
- Desktop app
  - Probably Electron because I know web tech, don't judge me
- Web interface
  - Easier to maintain than desktop apps
- Mobile apps
  - React Native or Flutter, haven't decided
- Browser extension
  - Quick AI access from any webpage

## Cool But Probably Won't Happen
- Voice input/output
  - Would need speech recognition and TTS APIs
- Real-time collaboration
  - Multiple people in same chat session, complex state management
- Conversation branching
  - Like git for chats, save/load different conversation states
- Plugin system
  - Let others write custom engines or commands
- Integration with GitHub/Slack/whatever
  - Webhooks and API integrations everywhere
- Local model support
  - Run Ollama locally for the paranoid

## Performance Stuff
- Response caching
  - Save identical responses to reduce API costs
- Better retry logic
  - Exponential backoff instead of just failing
- Connection pooling
  - Reuse HTTP connections for better performance

## Business Ideas (If This Gets Popular)
- Usage tracking
  - See which models cost the most, optimize spending
- Premium subscriptions
  - Pay monthly, use my API keys instead of yours
- Team management
  - Shared sessions and billing for companies
- Admin dashboards
  - Monitor usage, costs, popular features

## Technical Debt
- Write actual tests
  - I know, I know... but the codebase is getting big
- Docker containers
  - Easy deployment and development environments
- Proper CI/CD
  - Auto-deploy when I push to main
- Documentation that isn't just the README
  - API docs, architecture guides, all that boring stuff

## Wild Ideas
- AI that suggests better prompts
  - Meta-AI that teaches you to prompt better
- Conversation analytics
  - Which topics, models, prompts work best
- Multi-modal everything
  - Text + images + voice + video in one conversation
- Integration with note-taking apps
  - Auto-save interesting responses to Notion/Obsidian
- AI-powered debugging helper
  - Upload error logs, get suggested fixes
- Blockchain something something
  - Just kidding, but maybe conversation provenance?

*Half of this will never happen, but hey, gotta dream big.*
