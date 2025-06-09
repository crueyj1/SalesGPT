from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from salesgpt.prompts import SALES_AGENT_INCEPTION_PROMPT, STAGE_ANALYZER_INCEPTION_PROMPT


class SalesConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        sales_agent_inception_prompt = SALES_AGENT_INCEPTION_PROMPT
        prompt = PromptTemplate(
            template=sales_agent_inception_prompt,
            input_variables=[
                "salesperson_name",
                "salesperson_role", 
                "company_name",
                "company_business",
                "company_values",
                "conversation_purpose",
                "conversation_type",
                "conversation_history",
                "tools",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the agent move to next."""

    @classmethod  
    def from_llm(cls, llm, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        stage_analyzer_inception_prompt = STAGE_ANALYZER_INCEPTION_PROMPT
        prompt = PromptTemplate(
            template=stage_analyzer_inception_prompt,
            input_variables=["conversation_history", "current_stage"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
