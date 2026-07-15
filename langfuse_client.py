"""langfuse_client.py — Langfuse client, kept on its own OpenTelemetry TracerProvider.

Logfire also registers on the global TracerProvider; if Langfuse did the same, its
default export filter would drop our manual Logfire-created spans (they lack
`gen_ai.*` attributes), and Logfire's PII scrubbing could redact Langfuse's own
session/trace IDs. An isolated TracerProvider keeps the two fully separate.
"""

from dotenv import load_dotenv
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider

load_dotenv()

langfuse = Langfuse(tracer_provider=TracerProvider(), environment="local")
