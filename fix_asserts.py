file_path = "c:/Users/prabh/Desktop/LLM_Personal/personal_llm/api.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    'assert model_manager is not None, "Model manager not initialized"',
    'if model_manager is None:\n        raise HTTPException(status_code=503, detail="Model manager not initialized")'
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

count = sum(1 for l in lines if "assert " in l and "was asserted" not in l)
print(f"Remaining assert statements: {count}")