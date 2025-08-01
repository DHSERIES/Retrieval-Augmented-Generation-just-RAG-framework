from langchain.llms.base import LLM
from langchain.schema import LLMResult


#1 Use LangChain’s Chat Model Adapters
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#2 Use LangChain’s Text-only LLM Adapters
# llm = OpenAI(model="text-davinci-003", temperature=0)

#3 


class MyLLM(LLM):
    @property
    def _identifying_params(self):
        return {"model": "my-custom"}

    def _call(self, prompt: str, **kwargs) -> str:
        # call your API/inference here
        return my_api.generate_text(prompt)

    def _generate(self, prompts: list[str], **kwargs) -> LLMResult:
        gens = []
        for p in prompts:
            text = self._call(p, **kwargs)
            gens.append([{"text": text}])
        return LLMResult(generations=gens)