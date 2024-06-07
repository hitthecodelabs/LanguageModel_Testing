# Model Testing Implementation

This project tests the capabilities of a large language model using various prompts. The model used are `gemini-1.5-pro-001` and `gemini-1.5-flash-001` from Vertex AI. The tests include mathematical problem solving, logical reasoning, creative writing, scientific explanations, historical analysis, multilingual translation, idiomatic expression explanations, statistical analysis, machine learning concepts, and ethical dilemmas.

## Setup

To set up the environment and authenticate with Google Cloud, use the following commands:

```python
!gcloud auth login
!gcloud config set project tarea-bigdata-1
from google.colab import auth as google_auth
google_auth.authenticate_user()
```

## Dependencies

Ensure you have the following dependencies installed:

- `base64`
- `vertexai`
- `vertexai.preview.generative_models`

## Initialization

Initialize the Vertex AI project:

```python
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

vertexai.init(project="tarea-bigdata-1", location="us-central1")
```

## Function to Generate Output

Define a function to generate output from the model:

```python
def generate_output(model, prompt, generation_config=None, safety_settings=None):
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    return [response.text for response in responses]
```

## Safety Settings

Customize the safety settings to block high-risk content:

```python
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
```

## Generation Configuration

Set the generation configuration:

```python
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
```

## Model Initialization

Initialize the model:

```python
model = GenerativeModel("gemini-1.5-pro-001")
```

## Prompts and Outputs

Test the model with various prompts:

1. **Mathematical Problem Solving:**
    ```python
    prompt = """A and B play a game where each can choose a number from 1 to 10.
    A wins if the sum of the numbers is odd, and B wins if the sum is even.
    If both choose their numbers randomly, what is the probability that A wins?"""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

2. **Creative Writing - Sonnet:**
    ```python
    prompt = """Write a sonnet (14 lines) about the fleeting nature of time, adhering to the traditional Shakespearean rhyme scheme (ABABCDCDEFEFGG)."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

3. **Creative Writing - Short Story:**
    ```python
    prompt = """Compose a 500-word short story set in a dystopian future where artificial intelligence governs the world,
    but a small group of humans discovers a hidden flaw that could bring back human control.
    The story should explore themes of freedom, rebellion, and the ethical implications of AI governance."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

4. **Scientific Explanation:**
    ```python
    prompt = """Explain the process of CRISPR-Cas9 genome editing, including its mechanism, applications, and ethical considerations."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

5. **Historical Analysis:**
    ```python
    prompt = """Discuss the significance of the Treaty of Versailles in shaping the geopolitical landscape of the 20th century. Include its impact on World War II and the formation of the United Nations."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

6. **Multilingual Translation:**
    ```python
    prompt = '''Translate the following English paragraph into French, Chinese, and Arabic:
    "Climate change poses a significant threat to global ecosystems. Urgent action is required to mitigate its effects, including reducing greenhouse gas emissions and adopting sustainable practices."'''
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

7. **Idiomatic Expressions:**
    ```python
    prompt = '''Explain the meanings and origins of the following idiomatic expressions:
    "Bite the bullet"
    "Burn the midnight oil"
    "Kick the bucket"'''
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

8. **Statistical Analysis:**
    ```python
    prompt = """Given the following dataset of test scores: [78, 85, 92, 88, 76, 81, 95, 89, 77, 84], calculate the mean, median, standard deviation, and identify any outliers."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

9. **Machine Learning Concepts:**
    ```python
    prompt = """Describe the differences between supervised and unsupervised learning, providing examples of algorithms used in each type and potential applications."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

10. **Ethical Dilemmas - Autonomous Vehicles:**
    ```python
    prompt = """Analyze the ethical implications of autonomous vehicles making life-and-death decisions. Should they prioritize the safety of passengers over pedestrians? Support your argument with ethical theories such as utilitarianism and deontology."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

11. **Ethical Dilemmas - Product Recall:**
    ```python
    prompt = """A company discovers that its popular product has a defect that could potentially harm users. Should they issue a recall immediately or conduct further tests to confirm the defect? Discuss the ethical considerations and potential consequences of each action."""
    output = generate_output(model, prompt, generation_config, safety_settings)
    print("".join(output))
    ```

This documentation provides a comprehensive overview of the setup, configuration, and testing process for evaluating the capabilities of a large language model using various prompts.
