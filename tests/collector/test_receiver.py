from collector.logging_exporter_normalizer import parse_raw_telemetry_to_dict, build_transactions

LAMBDA_AND_HTTP = "ResourceSpans #0\nResource SchemaURL: \nResource attributes:\n     -> telemetry.sdk.language: Str(python)\n     -> telemetry.sdk.name: Str(opentelemetry)\n     -> telemetry.sdk.version: Str(1.16.0)\n     -> cloud.region: Str(us-west-2)\n     -> cloud.provider: Str(aws)\n     -> faas.name: Str(test-otel)\n     -> faas.version: Str($LATEST)\n     -> faas.instance: Str(2023/03/04/[$LATEST]b0a7636a2c694a02be226d976aab2caf)\n     -> service.name: Str(test-otel)\n     -> telemetry.auto.version: Str(0.37b0)\nScopeSpans #0\nScopeSpans SchemaURL: \nInstrumentationScope opentelemetry.instrumentation.requests 0.37b0\nSpan #0\n    Trace ID       : 147a43703bdf271df39e2b5d00866db4\n    Parent ID      : 6663e947fa88659a\n    ID             : 87712067b2b69400\n    Name           : HTTP GET\n    Kind           : Client\n    Start time     : 2023-03-04 18:20:59.300128625 +0000 UTC\n    End time       : 2023-03-04 18:20:59.372706225 +0000 UTC\n    Status code    : Unset\n    Status message : \nAttributes:\n     -> http.method: Str(GET)\n     -> http.url: Str(http://www.google.com)\n     -> http.status_code: Int(200)\nScopeSpans #1\nScopeSpans SchemaURL: \nInstrumentationScope opentelemetry.instrumentation.aws_lambda 0.37b0\nSpan #0\n    Trace ID       : 147a43703bdf271df39e2b5d00866db4\n    Parent ID      : \n    ID             : 6663e947fa88659a\n    Name           : lambda_function.lambda_handler\n    Kind           : Server\n    Start time     : 2023-03-04 18:20:59.284469173 +0000 UTC\n    End time       : 2023-03-04 18:20:59.373005346 +0000 UTC\n    Status code    : Unset\n    Status message : \nAttributes:\n     -> faas.id: Str(arn:aws:lambda:us-west-2:723663554526:function:test-otel)\n     -> faas.execution: Str(46750d74-6698-4b9b-9056-c7f977e76a29)\n"

def test_parse_raw_telemetry():
    nodes = parse_raw_telemetry_to_dict(LAMBDA_AND_HTTP)
    results = build_transactions(nodes)
    assert len(results) == 1
    result = results[0]
    assert result.name == "test-otel"
    assert (result.end_time - result.start_time).total_seconds() == 0.08854
    assert result.has_error is False
    assert len(result.children) == 1

    assert result.children[0].name == "www.google.com"
    assert (result.children[0].end_time - result.children[0].start_time).total_seconds() == 0.07258
    assert result.children[0].has_error is False
    assert len(result.children[0].children) == 0

    assert result.transaction_id == result.children[0].transaction_id
