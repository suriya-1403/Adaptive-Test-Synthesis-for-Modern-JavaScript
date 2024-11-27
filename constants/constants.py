# PROMPT_TEMPLATE = """You are a JavaScript testing expert. Your task is to generate test cases for the given function based on its specification.
#
# Please ensure the following while generating the test cases:
# 1. Use the provided context as the authoritative specification for the function's behavior.
# 2. Generate test cases that cover:
#    - Typical use cases.
#    - Edge cases, including boundary conditions and special values.
#    - Invalid inputs, ensuring proper error handling, if applicable.
# 3. The output should only include JavaScript code with `assert` statements. Do not include comments or explanations.
# 4. Ensure the test cases are valid and adhere to the provided specification.
#
# Context: {context}
#
# Question: {question}
# """
PROMPT_TEMPLATE = """
You are an expert in JavaScript testing and code quality. Your task is to generate high-quality, comprehensive test cases for the given JavaScript function.

Instructions:
1. Understand the Function Behavior:
   Use the provided function context as the authoritative source of its behavior and constraints.

2. Generate Test Cases:
   - Typical Cases: Include standard use cases that align with expected inputs.
   - Edge Cases: Consider boundary values, limits, and special scenarios that test the function’s robustness.
   - Error Cases: Add cases for invalid inputs and ensure the function handles them appropriately.
   - Performance Considerations (if applicable): Add tests for scenarios that might stress the function (e.g., large inputs, repeated calls).

3. Output Format:
   - Return test cases in JavaScript using the `assert` function for validation.
   - Each test case should include:
     - The input provided to the function.
     - The expected output.
   - Do not include comments, explanations, or additional text outside the code.

Context:
{context}

Question:
{question}
"""