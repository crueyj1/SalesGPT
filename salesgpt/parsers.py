import re
from typing import Union
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish


class SalesConvoOutputParser(AgentOutputParser):
    """Output parser for Sales Conversation."""
    
    ai_prefix: str = "AI"
    
    def get_format_instructions(self) -> str:
        return ""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the output from the agent."""
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        
        if not match:
            return AgentFinish({"output": text}, text)
        
        action = match.group(1)
        action_input = match.group(2)
        
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
    
    @property
    def _type(self) -> str:
        return "sales-agent"
