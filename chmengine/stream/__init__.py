"""Some silly idea to make the PS filewatcher invokable from python"""
try:
    from chmengine.stream.scripts import run_stream_alerts
except NotImplementedError as no_imp_error:
    raise NotImplementedError("chmengine.stream has no Python support at this time.") from no_imp_error

# pylint: disable=unreachable
__all__ = ['run_stream_alerts']
