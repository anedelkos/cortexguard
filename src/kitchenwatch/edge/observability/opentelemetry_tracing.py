from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

OPENTELEMETRY_ENDPOINT = "http://tempo:4318/v1/traces"
SERVICE_NAME = "CortexGuard-edge"


def setup_opentelemetry_tracing() -> None:

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    exporter = OTLPSpanExporter(endpoint=OPENTELEMETRY_ENDPOINT)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
