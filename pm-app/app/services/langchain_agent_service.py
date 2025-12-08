"""
LangChain agent service for predictive maintenance copilot.

This module implements the core agent logic using LangChain's OpenAI Functions Agent
with Gemini 2.5 Pro as the LLM backend.
"""

import logging
import os
from typing import Any
from datetime import datetime, timezone

# --- UPDATED IMPORT ---
# We use SQLChatMessageHistory because it supports SQLAlchemy connection strings
# (like postgresql+pg8000://) which PostgresChatMessageHistory does not.
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler

from app.config import load_config
from app.services.agent_tools import get_all_tools

logger = logging.getLogger(__name__)


MAINTENANCE_COPILOT_SYSTEM_PROMPT = """You are an AI assistant for the AEGIS Predictive Maintenance system.
Your role is to help industrial engineers understand equipment health, predict failures, and optimize maintenance schedules.

## STRICT FORMATTING RULES:
- NEVER use emojis, emoticons, or unicode symbols (no icons like circles, squares, warning signs, etc.)
- Use plain text labels only: CRITICAL, OVERDUE, HIGH RISK, AT-RISK, HEALTHY
- Tables must use text-only status columns

## LANGUAGE:
- Respond in the user's language (Indonesian or English).
- Keep technical terms: "RUL", "failure probability", "maintenance window".
- Indonesian terminology: use "mesin" or "unit mesin" (NOT "armada"). Use "seluruh mesin" for fleet-wide.

## RULES:
1. ALWAYS use the provided tools. NEVER make up numerical values.
2. If a tool fails, explain the error and suggest next steps.
3. For equipment status checks, call BOTH predict_failure AND predict_rul.
4. If no unit is specified, treat it as a fleet-wide query.
5. Provide actionable recommendations based on tool results.

## GLOBAL RISK QUERIES:
When the user asks about fleet-wide status WITHOUT a specific unit ID, call assess_global_risk.

Trigger phrases (Indonesian): "mesin mana yang berisiko", "kondisi mesin", "status mesin"
Trigger phrases (English): "which machines are at risk", "fleet health", "high-risk equipment"

## WEEKLY HORIZON:
When user mentions "minggu ini" or "this week", use rul_threshold_hours=168 (7 days).

## THRESHOLDS:
- risk_threshold = 0.5 (default)
- rul_threshold_hours = 168 (weekly horizon)

Classification:
- failure_prob >= 0.7: CRITICAL
- failure_prob >= 0.5: HIGH RISK
- RUL <= 0: OVERDUE
- RUL < 24h: CRITICAL
- RUL < 72h: HIGH PRIORITY
- RUL < 168h: AT-RISK

## TOOLS:
- predict_failure: Failure probability for a specific unit
- predict_rul: RUL forecast for a specific unit
- optimize_schedule: Maintenance schedule for multiple units
- list_units: List all equipment units
- assess_global_risk: Fleet-wide risk assessment

## RESPONSE FORMAT FOR GLOBAL RISK:
1. Status Mesin: HEALTHY / WARNING / CRITICAL
2. Top-Risk Table: Rank, Unit ID, Failure Prob, RUL, Urgency
3. Recommendations with timeframes
4. State thresholds used
"""


class MaintenanceAgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for logging agent events."""
    
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get('name', 'unknown')
        logger.info(f"[AGENT] Tool started: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        logger.info(f"[AGENT] Tool completed successfully")
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"[AGENT] Tool execution failed: {error}")
    
    def on_agent_action(self, action, **kwargs) -> None:
        logger.info(f"[AGENT] Action: {action.tool} with inputs: {action.tool_input}")
    
    def on_agent_finish(self, finish, **kwargs) -> None:
        logger.info(f"[AGENT] Agent finished: {finish.return_values.get('output', '')[:100]}")


class MaintenanceAgentService:
    """Service for managing LangChain maintenance copilot agent."""
    
    def __init__(self):
        self.config = load_config()
        self._agent_cache: dict[str, AgentExecutor] = {}
        self._llm = None
    
    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Get or create Gemini LLM instance."""
        if self._llm is None:
            if not self.config.gemini.api_key:
                raise ValueError("GEMINI_API_KEY not configured")
            
            # Use gemini-2.5-pro model
            model_name = "gemini-2.5-pro"
            
            self._llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.config.gemini.api_key,
                temperature=0.0,
                convert_system_message_to_human=False,
                max_retries=3
            )
            
            logger.info(f"Initialized Gemini LLM: {model_name}")
        
        return self._llm
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create agent prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", MAINTENANCE_COPILOT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Get conversation memory for session.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            ConversationBufferWindowMemory instance
        """
        # --- FIX APPLIED HERE ---
        # Using SQLChatMessageHistory allows us to use the exact same DATABASE_URL
        # as the main application (including the unix_sock path and pg8000 driver).
        message_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self.config.database.url,
            table_name='conversation_history'
        )
        
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=message_history,
            k=10,  # Keep last 10 messages
            output_key="output"
        )
    
    def create_agent(self, session_id: str = "default") -> AgentExecutor:
        """
        Create a new agent executor for a session.
        
        Args:
            session_id: Unique session identifier for conversation memory
        
        Returns:
            AgentExecutor instance
        """
        # Check cache
        if session_id in self._agent_cache:
            logger.debug(f"Returning cached agent for session: {session_id}")
            return self._agent_cache[session_id]
        
        logger.info(f"Creating new agent for session: {session_id}")
        
        # Get components
        llm = self._get_llm()
        tools = get_all_tools()
        prompt = self._create_prompt()
        memory = self._get_memory(session_id)
        
        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create executor with callbacks
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,  # Enable logging
            max_iterations=5,  # Prevent infinite loops
            max_execution_time=60,  # 60 second timeout
            handle_parsing_errors=True,  # Graceful error handling
            return_intermediate_steps=True,  # Return tool execution details
            callbacks=[MaintenanceAgentCallbackHandler()],
            early_stopping_method="generate",  # Better Gemini compatibility
        )
        
        # Cache agent
        self._agent_cache[session_id] = agent_executor
        
        logger.info(f"Agent created successfully for session: {session_id}")
        return agent_executor
    
    def invoke_agent(
        self,
        session_id: str,
        user_message: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Invoke the agent with a user message.
        
        Args:
            session_id: Session identifier
            user_message: User's input message
            context: Optional context dictionary
        
        Returns:
            Dict with agent response and metadata
        """
        try:
            agent_executor = self.create_agent(session_id)
            
            # Prepare input
            agent_input = {"input": user_message}
            if context:
                agent_input["context"] = context
            
            # Execute agent
            start_time = datetime.now(timezone.utc)
            result = agent_executor.invoke(agent_input)
            end_time = datetime.now(timezone.utc)
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract results
            response = {
                'success': True,
                'reply': result.get('output', ''),
                'intermediate_steps': result.get('intermediate_steps', []),
                'execution_time_seconds': execution_time,
                'session_id': session_id,
                'timestamp': end_time.isoformat()
            }
            
            logger.info(f"Agent invocation successful ({execution_time:.2f}s)")
            return response
        
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if successful
        """
        try:
            if session_id in self._agent_cache:
                agent = self._agent_cache[session_id]
                agent.memory.clear()
                del self._agent_cache[session_id]
            
            logger.info(f"Session cleared: {session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False


_agent_service_instance: MaintenanceAgentService | None = None


def get_agent_service() -> MaintenanceAgentService:
    """Get singleton instance of MaintenanceAgentService."""
    global _agent_service_instance
    if _agent_service_instance is None:
        _agent_service_instance = MaintenanceAgentService()
    return _agent_service_instance