AVAILABLE_MODELS = {
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_vl": "Qwen2_VL"
}

def get_models(model_name, logger):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")
    
    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"src.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise