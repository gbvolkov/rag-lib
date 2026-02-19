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


def test_summarize_payload_includes_language_and_length_hint():
    captured_inputs = []

    def fake_llm_func(input_val):
        captured_inputs.append(input_val)
        return AIMessage(content="ok")

    fake_llm = RunnableLambda(fake_llm_func)
    summarizer = LLMTableSummarizer(llm=fake_llm, soft_max_chars=321)
    summarizer.prompt = RunnableLambda(lambda x: x)

    table_md = (
        "| \u041f\u043e\u043b\u0435 | \u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435 |\n"
        "|---|---|\n"
        "| \u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 | \u041f\u0440\u043e\u0434\u0443\u043a\u0442 |"
    )
    result = summarizer.summarize(table_md)

    assert result == "ok"
    assert len(captured_inputs) == 1
    invoked_input = captured_inputs[0]
    assert invoked_input["target_language"] == "russian"
    assert invoked_input["soft_max_chars"] == 321
    assert invoked_input["table_content"] == table_md
