def print_model_summary(model):
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
