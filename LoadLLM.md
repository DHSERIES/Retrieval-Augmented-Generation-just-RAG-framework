## Load from api 
example 
"""llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)"""

## Loading LLM with a local inference library (e.g., llama.cpp, ctransformers, or transformers from Hugging Face).
example
"""from langchain.llms import CTransformers

llm = CTransformers(
    model="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    max_new_tokens=512,
    temperature=0.7
)"""

## Load Your Custom Model with HuggingFace
example
- **from scratch**
"""from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "/path/to/your/local/model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False

from langchain.llms import HuggingFacePipeline
local_llm = HuggingFacePipeline(pipeline=pipe)
)"""
