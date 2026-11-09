"""Этап 10: трейсы OpenTelemetry.

Один запрос /ask — это один трейс со спанами retrieve → rerank → generate. Без
OTEL_EXPORTER_OTLP_ENDPOINT спаны считаются, но никуда не уходят (почти бесплатно);
с ним — улетают в Jaeger: docker compose --profile otel up -d, UI на http://localhost:16686.
"""

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

tracer = trace.get_tracer("rag-lab")


def setup_tracing(exporter: SpanExporter | None = None) -> TracerProvider:
    """Поднять провайдер один раз на процесс. exporter передают тесты (InMemorySpanExporter)."""
    provider = TracerProvider(resource=Resource.create({"service.name": "rag-lab"}))
    if exporter is None and os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()  # адрес берёт из переменной окружения
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider
