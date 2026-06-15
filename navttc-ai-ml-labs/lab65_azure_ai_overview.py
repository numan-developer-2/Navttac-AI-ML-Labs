# Lab 65: Microsoft Azure AI and ML services overview
# Study notes aligned with Microsoft Learn training paths.

print("-" * 48)
print("Lab 65: Microsoft Azure AI & ML Services")
print("-" * 48)

sections = [
    (
        "Azure platform overview",
        """
Azure is Microsoft's cloud platform with compute, storage, databases,
networking, and AI/ML services. Common building blocks include:
  - Virtual Machines and Azure Functions (compute)
  - Blob Storage and Data Lake (storage)
  - Azure SQL and Cosmos DB (databases)
  - Azure ML, Cognitive Services, and Azure OpenAI (AI)
""",
    ),
    (
        "Azure Machine Learning",
        """
Azure ML supports the full model lifecycle:
  workspace, compute clusters, datastores, datasets, pipelines,
  designer UI, and AutoML.

Typical workflow:
  ingest data -> prepare features -> train model -> evaluate -> deploy -> monitor
""",
    ),
    (
        "Cognitive Services",
        """
Pre-built APIs for vision, language, speech, and decision tasks:
  - Vision: Computer Vision, Face API, Custom Vision
  - Language: Text Analytics, Translator, LUIS
  - Speech: speech-to-text and text-to-speech
  - Decision: Anomaly Detector, Personalizer
""",
    ),
    (
        "Azure OpenAI Service",
        """
Hosts OpenAI models (GPT, DALL-E, Whisper) inside Azure security boundaries.
Use cases include summarization, code assistance, embeddings, and RAG apps.
""",
    ),
    (
        "Data services for ML",
        """
  - Azure Data Factory: ETL and orchestration
  - Azure Databricks: Spark-based analytics
  - Azure Synapse: analytics warehouse
  - Azure Data Lake: large-scale raw data storage
""",
    ),
]

for title, body in sections:
    print(f"\n[{title}]")
    print(body.strip())

sdk_example = '''
# Azure ML Python SDK v2 (requires azure-ai-ml + azure-identity)
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group_name="<resource-group>",
    workspace_name="<workspace-name>",
)

for target in client.compute.list():
    print(target.name, target.type)

training_job = command(
    code="./src",
    command="python train.py --lr 0.01 --epochs 20",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
    compute="cpu-cluster",
    display_name="sklearn-training-run",
)
submitted = client.jobs.create_or_update(training_job)
print("Submitted job:", submitted.name)
'''

print("\n[SDK example]")
print(sdk_example.strip())

print(
    """
[Certifications to explore on Microsoft Learn]
  - AI-900: Azure AI Fundamentals
  - AI-102: Azure AI Engineer
  - DP-100: Azure Data Scientist

[Cloud ML comparison]
  Azure ML  -> strong enterprise integration
  SageMaker -> AWS ecosystem
  Vertex AI -> Google Cloud research tooling

Complete the free modules at:
https://learn.microsoft.com/en-us/training/
"""
)
