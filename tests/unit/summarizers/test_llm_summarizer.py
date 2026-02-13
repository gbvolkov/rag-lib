import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from rag_lib.summarizers.table_llm import LLMTableSummarizer

def test_summarize_table():
    # Use a real RunnableLambda to simulate the LLM
    # This avoids MagicMock issues with LangChain's `|` operator coercion
    captured_inputs = []
    
    def fake_llm_func(input_val):
        captured_inputs.append(input_val)
        return AIMessage(content="This is a summary.")
    
    fake_llm = RunnableLambda(fake_llm_func)
    
    summarizer = LLMTableSummarizer(llm=fake_llm)
    # Override prompt to avoid ChatPromptTemplate complexity in unit test
    summarizer.prompt = RunnableLambda(lambda x: f"Prompt: {x['table_content']}")
    
    table_md = "| Header | \n | --- | \n | Value |"
    result = summarizer.summarize(table_md)
    
    # Check result
    assert result == "This is a summary."
    
    # Verify input contained the table
    assert len(captured_inputs) == 1
    invoked_input = captured_inputs[0]
    assert table_md in str(invoked_input)
