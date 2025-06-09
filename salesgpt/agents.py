import re
from typing import Dict, List, Any, Optional
from langchain.llms.base import BaseLLM
from langchain.agents import AgentExecutor
from langchain.schema import BaseMessage
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.prompts import SALES_AGENT_INCEPTION_PROMPT
from salesgpt.tools import get_tools
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.chains import SalesConversationChain, StageAnalyzerChain


class SalesGPT:
    """Controller model for the Sales Agent."""
    
    def __init__(
        self,
        stage_id: int = 1,
        salesperson_name: str = "Ted Lasso",
        salesperson_role: str = "Business Development Representative",
        company_name: str = "Sleep Haven",
        company_business: str = "Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible.",
        company_values: str = "Our mission at Sleep Haven is to help people achieve a better night's sleep by providing exceptional mattresses at an affordable price.",
        conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress.",
        conversation_type: str = "call",
        use_tools: bool = False,
        product_catalog: str = None,
        llm: BaseLLM = None,
        verbose: bool = False,
    ):
        self.stage_id = stage_id
        self.salesperson_name = salesperson_name
        self.salesperson_role = salesperson_role
        self.company_name = company_name
        self.company_business = company_business
        self.company_values = company_values
        self.conversation_purpose = conversation_purpose
        self.conversation_type = conversation_type
        self.use_tools = use_tools
        self.product_catalog = product_catalog
        self.llm = llm
        self.verbose = verbose
        
        self.conversation_history = []
        self.current_stage = CONVERSATION_STAGES.get(stage_id, CONVERSATION_STAGES[1])
        
        # Initialize chains
        self.sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm=self.llm, verbose=self.verbose
        )
        
        self.stage_analyzer_chain = StageAnalyzerChain.from_llm(
            llm=self.llm, verbose=self.verbose
        )
        
        # Initialize tools if needed
        if self.use_tools:
            self.sales_agent_executor = self._get_agent_executor()
    
    def seed_agent(self):
        """Seed the agent with initial conversation stage."""
        self.current_stage = CONVERSATION_STAGES[self.stage_id]
        self.conversation_history = []
    
    def determine_conversation_stage(self):
        """Determine current conversation stage based on history."""
        if not self.conversation_history:
            return "1"
            
        stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history),
            current_stage=self.current_stage,
        )
        
        self.stage_id = int(stage_id)
        self.current_stage = CONVERSATION_STAGES.get(self.stage_id, CONVERSATION_STAGES[1])
        
        return stage_id
    
    def step(self):
        """Run one step of the sales conversation."""
        return self._call(inputs={})
    
    def _call(self, inputs: Dict[str, Any]):
        """Run the sales agent."""
        if self.use_tools:
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
                product_catalog=self.product_catalog,
            )
        else:
            ai_message = self.sales_conversation_utterance_chain.run(
                conversation_stage=self.current_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
                product_catalog=self.product_catalog,
            )
        
        # Clean the AI message
        agent_name = self.salesperson_name
        ai_message = ai_message.replace(f"{agent_name}: ", "")
        if not ai_message.startswith(agent_name):
            ai_message = f"{agent_name}: {ai_message}"
        
        self.conversation_history.append(ai_message)
        
        if self.verbose:
            print(f"({self.salesperson_name}): {ai_message}")
            
        return ai_message
    
    def human_step(self, human_input: str):
        """Process human input and add to conversation history."""
        human_input = f"User: {human_input}"
        self.conversation_history.append(human_input)
    
    def _get_agent_executor(self) -> AgentExecutor:
        """Get agent executor with tools."""
        tools = get_tools(self.product_catalog)
        
        prompt = SALES_AGENT_INCEPTION_PROMPT
        
        llm_chain = SalesConversationChain.from_llm(
            llm=self.llm, 
            verbose=self.verbose
        )
        
        tool_names = [tool.name for tool in tools]
        
        sales_agent_with_tools = SalesGPTWithTools(
            llm_chain=llm_chain,
            tools=tools,
            verbose=self.verbose,
            output_parser=SalesConvoOutputParser(ai_prefix=self.salesperson_name),
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=sales_agent_with_tools,
            tools=tools,
            verbose=self.verbose,
        )


class SalesGPTWithTools:
    """Sales agent with tools capability."""
    
    def __init__(self, llm_chain, tools, verbose=False, output_parser=None):
        self.llm_chain = llm_chain
        self.tools = tools
        self.verbose = verbose
        self.output_parser = output_parser
    
    def plan(self, intermediate_steps, **kwargs):
        """Plan next action."""
        return self.llm_chain.run(**kwargs)
