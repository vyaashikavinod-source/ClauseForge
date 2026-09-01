"""Controlled service errors safe for public responses."""


class ServingError(Exception):
    status_code = 500
    code = "internal_error"
    public_message = "An internal processing error occurred."


class ProviderUnavailableError(ServingError):
    status_code = 503
    code = "provider_unavailable"
    public_message = "The classification provider is unavailable."


class InvalidModelOutputError(ServingError):
    status_code = 422
    code = "invalid_model_output"
    public_message = "The model did not produce a valid taxonomy category."


class InferenceTimeoutError(ServingError):
    status_code = 504
    code = "inference_timeout"
    public_message = "Classification exceeded the configured time limit."


class InputTooLargeError(ServingError):
    status_code = 413
    code = "input_too_large"
    public_message = "The clause text exceeds the configured input limit."
