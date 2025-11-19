import json
from app.models.response_models import AgentResponse

print(json.dumps(AgentResponse.model_json_schema(), indent=2))
