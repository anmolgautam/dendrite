## What is Dendrite?
This started as a personal project to learn Agent Building and platform development to understand the ecosystem completely.

## Formal Definition
 
 Dendrite is a backend platform that provides a durable, multi-agent, LLM-powered workflow engine with tool orchestration, agent hierarchy, and a unified execution model for client tools, backend tools, and built-in Dendrite tools.

## Dendrite has following Primitives : 
1. Agent - An LLM-powered controller that runs a ReAct loop and can call tools or subagents.

2. Root Agent - The main orchestrator responsible for the entire workflow in a conversation.

3. Agent-as-Tool (Subagent) - A secondary agent invoked like a tool, with its own ReAct loop and delegation level.

4. Tool - A callable function an agent can invoke during execution.

5. Client Tool - A tool that executes in the application’s frontend (e.g., Excel, browser UI).

6. Backend Tool - A tool that executes in the developer’s backend (API calls, DB queries, business logic).

7. Dendrite Native Tool - A tool implemented inside Dendrite itself (file search, web search, RAG, parsing).

8. Conversation - A long-lived context that stores messages, files, and agent runs for an end-user.

9. Message - A user or agent utterance that forms part of the conversation history.

10. Run (Agent Run) - A single execution of an agent triggered by a new user message.

11. Tool Call - A structured request from an agent to execute a tool, recorded durably with status.

12. Reasoning Trace - A step-by-step log of the agent’s internal reasoning and decisions during a run.

13. Delegation Level - The depth of subagent invocation (0 = root, 1 = subagent, 2 = sub-subagent, …).

14. Instance ID - A unique identifier for each subagent invocation within a run.

15. Workspace - An isolated tenant environment where agents, tools, conversations, and runs live.

16. API Key - A credential tied to a workspace used by the app backend to access Dendrite.

17. Dendrite SDK - A client library used to register tools, start runs, and handle tool execution.

18. Worker - A backend process that consumes backend tool calls and sends results to Dendrite.

19. Runtime - The Dendrite execution engine that drives ReAct loops, tool orchestration, and agent hierarchy.

20. Stream Queue - A Redis-based event pipeline used for dispatching backend tool calls and resuming workflows.